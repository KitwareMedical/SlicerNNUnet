from typing import Optional

import qt
import slicer

from .InstallLogic import InstallLogic
from .IconPath import icon
from .SegmentationLogic import SegmentationLogic, SegmentationLogicProtocol
from .Utils import (
    createButton, addInCollapsibleLayout,
)


class Widget(qt.QWidget):
    def __init__(self, segmentationLogic: Optional[SegmentationLogicProtocol] = None, installLogic=None, parent=None):
        import ctk
        super().__init__(parent)
        self.logic = segmentationLogic or SegmentationLogic()
        self.installLogic = installLogic or InstallLogic()
        self.nnUnetPackageName = "nnunetv2"

        # Volume node input selector
        self.inputSelector = slicer.qMRMLNodeComboBox(self)
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)

        # nnUNet model path selector
        path_form = qt.QFormLayout()
        self.nnUnetModelPath = ctk.ctkPathLineEdit()

        # Settings widget
        self.currentVersionLabel = qt.QLabel("")
        self.toInstallLineEdit = qt.QLineEdit()
        self.toInstallLineEdit.placeholderText = ">=2.0.0"
        self.installButton = qt.QPushButton("Install")
        self.installButton.clicked.connect(self.onApply)

        settingsWidget = qt.QWidget()
        settingsLayout = qt.QFormLayout(settingsWidget)
        settingsLayout.addRow(qt.QLabel("nnUNet version"))
        settingsLayout.addRow("Current:", self.currentVersionLabel)
        settingsLayout.addRow("To install:", self.toInstallLineEdit)
        settingsLayout.addRow(self.installButton)

        # Segmentation apply / stop buttons
        self.applyButton = createButton(
            "Apply",
            callback=self.onApplyClicked,
            toolTip="Click to run the segmentation.",
            icon=icon("start_icon.png")
        )

        self.stopButton = createButton(
            "Stop",
            callback=self.onStopClicked,
            toolTip="Click to Stop the segmentation.",
            icon=icon("stop_icon.png")
        )

        self.logTextEdit = qt.QTextEdit()
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setLineWrapMode(qt.QTextEdit.NoWrap)

        # Widget layout
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.inputSelector)
        addInCollapsibleLayout(settingsWidget, layout, "nnUNet settings")
        layout.addWidget(self.applyButton)
        layout.addWidget(self.stopButton)
        layout.addWidget(self.logTextEdit)
        layout.addStretch()

        # Logic connection
        self.logic.inferenceFinished.connect(self.onInferenceFinished)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.installLogic.progressInfo.connect(self.onProgressInfo)
        self.isStopping = False

        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self.onInputChanged()
        self._setApplyVisible(True)

    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def onInstall(self):
        self.installButton.setEnabled(False)
        self.applyButton.setEnabled(False)
        self.stopButton.setEnabled(False)

        with slicer.util.tryWithErrorDisplay("Failed to install nnUNetV2", show=True, waitCursor=True):
            try:
                self.logTextEdit.clear()
                self.installLogic.setupPythonRequirements(self.nnUnetPackageName + self.toInstallLineEdit.text)
                slicer.util.infoDisplay("Install finished correctly.")
            finally:
                self.installButton.setEnabled(True)
                self.applyButton.setEnabled(True)
                self.stopButton.setEnabled(True)
                self.updateInstalledVersion()

    def onLogMessage(self, msg):
        self.logTextEdit.insertPlainText(msg + "\n")

    def updateInstalledVersion(self):
        self.currentVersionLabel.setText(str(self.installLogic.getInstalledPackageVersion(self.nnUnetPackageName)))

    def onSceneChanged(self, *_):
        self.onStopClicked()

    def onStopClicked(self):
        self.isStopping = True
        self.logic.stopSegmentation()
        self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    def onApplyClicked(self, *_):
        self.logTextEdit.clear()
        self.onProgressInfo("Start")
        self.onProgressInfo("*" * 80)
        self._setApplyVisible(False)
        self._runSegmentation()

    def _setApplyVisible(self, isVisible):
        self.applyButton.setVisible(isVisible)
        self.stopButton.setVisible(not isVisible)
        self.inputSelector.setEnabled(isVisible)

    def _runSegmentation(self):
        if not self.installLogic.isPackageInstalledAndCompatible(self.nnUnetPackageName):
            self.onInstall()

        if self.installLogic.needsRestart:
            self.onInferenceFinished()
            return

        self.logic.startSegmentation(self.getCurrentVolumeNode())

    def onInputChanged(self, *_):
        self.applyButton.setEnabled(self.getCurrentVolumeNode() is not None)

    def getCurrentVolumeNode(self):
        return self.inputSelector.currentNode()

    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return

        try:
            self.onProgressInfo("Loading inference results...")
            self.logic.loadSegmentation()
            self.onProgressInfo("Inference ended successfully.")
        except RuntimeError as e:
            slicer.util.errorDisplay(e)
            self.onProgressInfo(f"Error loading results :\n{e}")
        finally:
            self._setApplyVisible(True)

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return

        self._setApplyVisible(True)
        slicer.util.errorDisplay("Encountered error during inference :\n" + errorMsg)

    def onProgressInfo(self, infoMsg):
        self.logTextEdit.insertPlainText(infoMsg + "\n")
        self.moveTextEditToEnd(self.logTextEdit)
        slicer.app.processEvents()

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)
