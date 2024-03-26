import traceback
from pathlib import Path
from typing import Optional

import qt
import slicer
from slicer.parameterNodeWrapper import parameterNodeWrapper

from .InstallLogic import InstallLogic, InstallLogicProtocol
from .Parameter import Parameter
from .SegmentationLogic import SegmentationLogic, SegmentationLogicProtocol


@parameterNodeWrapper
class WidgetParameterNode:
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    parameter: Parameter


class Widget(qt.QWidget):
    """
    nnUNet widget containing an install and run settings collapsible areas.
    Allows to run nnUNet model and displays results in the UI.
    Saves the used settings to QSettings for reloading.
    """

    def __init__(
            self,
            segmentationLogic: Optional[SegmentationLogicProtocol] = None,
            installLogic: Optional[InstallLogicProtocol] = None,
            doShowInfoWindows: bool = True,
            parent=None
    ):
        super().__init__(parent)
        self.logic = segmentationLogic or SegmentationLogic()
        self.installLogic = installLogic or InstallLogic()

        # Instantiate widget UI
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        uiWidget = slicer.util.loadUI(self.resourcePath().joinpath("UI/SlicerNNUNet.ui").as_posix())
        uiWidget.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.inputSelector.currentNodeChanged.connect(self.onInputChanged)
        self.ui.installButton.clicked.connect(self.onInstall)

        self.ui.applyButton.setIcon(self.icon("start_icon.png"))
        self.ui.applyButton.clicked.connect(self.onApply)

        self.ui.stopButton.setIcon(self.icon("stop_icon.png"))
        self.ui.stopButton.clicked.connect(self.onStopClicked)

        # Logic connection
        self.logic.inferenceFinished.connect(self.onInferenceFinished)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.installLogic.progressInfo.connect(self.onProgressInfo)
        self.isStopping = False

        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self._doShowErrorWindows = doShowInfoWindows

        # Create parameter node and connect GUI
        self._parameterNode = self._createParameterNode()
        self._parameterNode.parameter = Parameter.fromSettings()
        self._parameterNode.connectParametersToGui(
            {
                "parameter.modelPath": self.ui.nnUNetModelPathEdit,
                "parameter.device": self.ui.deviceComboBox,
                "parameter.stepSize": self.ui.stepSizeSlider,
                "parameter.checkPointName": self.ui.checkPointNameLineEdit,
                "parameter.folds": self.ui.foldsLineEdit,
                "parameter.nProcessPreprocessing": self.ui.nProcessPreprocessingSpinBox,
                "parameter.nProcessSegmentationExport": self.ui.nProcessSegmentationExportSpinBox,
                "parameter.disableTta": self.ui.disableTtaCheckBox
            }
        )

        # Configure UI
        self.onInputChanged()
        self.updateInstalledVersion()
        self._setApplyVisible(True)

    @staticmethod
    def _createParameterNode() -> WidgetParameterNode:
        moduleName = "SlicerNNUNet"
        parameterNode = slicer.mrmlScene.GetSingletonNode(moduleName, "vtkMRMLScriptedModuleNode")
        if not parameterNode:
            parameterNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            parameterNode.SetName(slicer.mrmlScene.GenerateUniqueName(moduleName))

        parameterNode.SetAttribute("ModuleName", moduleName)
        return WidgetParameterNode(parameterNode)

    @staticmethod
    def resourcePath() -> Path:
        return Path(__file__).parent.joinpath("..", "Resources")

    @classmethod
    def icon(cls, icon_name) -> "qt.QIcon":
        return qt.QIcon(cls.resourcePath().joinpath("Icons", icon_name).as_posix())

    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def _setButtonsEnabled(self, isEnabled):
        self.ui.installButton.setEnabled(isEnabled)
        self.ui.applyButton.setEnabled(isEnabled)
        self.ui.inputSelector.setEnabled(isEnabled)

    def onInstall(self, *, doReportFinished=True):
        self._setButtonsEnabled(False)

        if doReportFinished:
            self.ui.logTextEdit.clear()

        success = self.installLogic.setupPythonRequirements(f"nnunetv2{self.ui.toInstallLineEdit.text}")
        if doReportFinished:
            if success:
                self._reportFinished("Install finished correctly.")
            else:
                self._reportError("Install failed.")
        self.updateInstalledVersion()
        self._setButtonsEnabled(True)
        return success

    def _reportError(self, msg, doTraceback=True):
        self.onProgressInfo(msg)
        if self._doShowErrorWindows:
            all_msgs = (msg,) if not doTraceback else (msg, traceback.format_exc())
            slicer.util.errorDisplay(*all_msgs)

    def _reportFinished(self, msg):
        self.onProgressInfo("*" * 80)
        self.onProgressInfo(msg)
        if self._doShowErrorWindows:
            slicer.util.infoDisplay(msg)

    def onLogMessage(self, msg):
        self.ui.logTextEdit.insertPlainText(msg + "\n")

    def updateInstalledVersion(self):
        self.ui.currentVersionLabel.setText(str(self.installLogic.getInstalledNNUnetVersion()))

    def onSceneChanged(self, *_):
        self.onStopClicked()

    def onStopClicked(self):
        self.isStopping = True
        self.logic.stopSegmentation()
        self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    def onApply(self, *_):
        if self.getCurrentVolumeNode() is None:
            self._reportError("Please select a valid volume to proceed.")
            return

        self.ui.logTextEdit.clear()
        self.onProgressInfo("Start")
        self.onProgressInfo("*" * 80)

        if not self.onInstall(doReportFinished=False):
            return

        if self.installLogic.needsRestart:
            self._reportFinished("Please restart 3D Slicer to proceed with segmentation.")
            return

        self._setApplyVisible(False)
        self._runSegmentation()

    def _setApplyVisible(self, isVisible):
        self.ui.applyButton.setVisible(isVisible)
        self.ui.stopButton.setVisible(not isVisible)
        self._setButtonsEnabled(isVisible)

    def _runSegmentation(self):
        if self.installLogic.needsRestart:
            self.onInferenceFinished()
            return

        self._parameterNode.parameter.toSettings()
        self.logic.setParameter(self._parameterNode.parameter)
        self.logic.startSegmentation(self.getCurrentVolumeNode())

    def onInputChanged(self, *_):
        self.ui.applyButton.setEnabled(self.getCurrentVolumeNode() is not None)

    def getCurrentVolumeNode(self):
        return self.ui.inputSelector.currentNode()

    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return

        try:
            self.onProgressInfo("Loading inference results...")
            segmentation = self.logic.loadSegmentation()
            segmentation.SetName(self.getCurrentVolumeNode().GetName() + "Segmentation")
            self._reportFinished("Inference ended successfully.")
        except RuntimeError as e:
            self._reportError(f"Inference ended in error:\n{e}")
        finally:
            self._setApplyVisible(True)

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return

        self._setApplyVisible(True)
        self._reportError("Encountered error during inference :\n" + errorMsg, doTraceback=False)

    def onProgressInfo(self, infoMsg):
        self.ui.logTextEdit.insertPlainText(self._formatMsg(infoMsg) + "\n")
        self.moveTextEditToEnd(self.ui.logTextEdit)
        slicer.app.processEvents()

    @staticmethod
    def _formatMsg(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines()])

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)
