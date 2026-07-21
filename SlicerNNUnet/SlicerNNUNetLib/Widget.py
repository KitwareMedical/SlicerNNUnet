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

        self.additionalInputSelectors = []

        # Instantiate widget UI
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        uiWidget = slicer.util.loadUI(self.resourcePath().joinpath("UI/SlicerNNUNet.ui").as_posix())
        uiWidget.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.inputSelector.currentNodeChanged.connect(self.onInputsChanged)
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

        self.ui.nnUNetModelPathEdit.connect("currentPathChanged(QString)", self.onModelPathChanged)

        # Configure UI
        self.updateInstalledVersion()
        self._setApplyVisible(True)

        self.onModelPathChanged()

    def onUpdateAdditionalInputSelectors(self, channels):
        formLayout = self.ui.additionalInputsFrame.layout()
        if len(self.additionalInputSelectors) > 0:
            for i in reversed(range(formLayout.rowCount())):
                formLayout.removeRow(i)
            self.additionalInputSelectors.clear()

        for idx, channel in enumerate(channels):
            if idx == 0:
                continue
            inputSelector = slicer.qMRMLNodeComboBox()
            inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            inputSelector.noneEnabled = True
            inputSelector.showHidden = False
            inputSelector.renameEnabled = True
            inputSelector.setMRMLScene(slicer.mrmlScene)
            inputSelector.currentNodeChanged.connect(self.onInputsChanged)
            formLayout.addRow(f"Input volume {idx + 1}: ", inputSelector)
            self.additionalInputSelectors.append(inputSelector)

    def onModelPathChanged(self):
        nnUNetParam = self._parameterNode.parameter
        isValid, messages = nnUNetParam.isValid()
        self.ui.logTextEdit.clear()
        if messages:
            self.onLogMessage(messages)
        self.ui.applyButton.setEnabled(False)

        channels = []
        if isValid:
            channels = nnUNetParam.readChannelNamesFromFile()
        self.onUpdateAdditionalInputSelectors(channels)
        self.onInputsChanged()

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
        if not self.getCurrentVolumeNodes():
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
        self.logic.startSegmentation(self.getCurrentVolumeNodes())

    def onInputsChanged(self, *_):
        self.ui.applyButton.setEnabled(len(self.getCurrentVolumeNodes()) > 0)

    def getCurrentVolumeNodes(self):
        return [self.ui.inputSelector.currentNode()] + [s.currentNode() for s in self.additionalInputSelectors]

    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return

        try:
            self.onProgressInfo("Loading inference results...")
            segmentation = self.logic.loadSegmentation()
            segmentation.SetName(self.ui.inputSelector.currentNode().GetName() + "Segmentation")
            self._reportFinished("Inference ended successfully.")
        except RuntimeError as e:
            self._reportError(f"Inference ended in error:\n{e}")
        finally:
            self._setApplyVisible(True)

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return

        self._setApplyVisible(True)
        if isinstance(errorMsg, Exception):
            errorMsg = str(errorMsg)
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
