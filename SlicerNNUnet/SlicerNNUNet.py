from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _, translate

from SlicerNNUNetLib import Widget


class SlicerNNUNet(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("nnUNet")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Thibault Pelletier (Kitware SAS)"]
        self.parent.helpText = _(
            "This extension is meant to streamline the integration of nnUnet based models into 3D Slicer.<br>"
            "It allows for quick and reliable nnUNet dependency installation in 3D Slicer environment and provides"
            " simple logic to launch nnUNet prediction on given directories.<br><br>"
            "The installation steps are based on the work done in the "
            '<a href="https://github.com/lassoan/SlicerTotalSegmentator/">Slicer Total Segmentator extension</a>'
        )
        self.parent.acknowledgementText = _(
            "This module was originally co-financed by the "
            '<a href="https://orthodontie-ffo.org/">Fédération Française d\'Orthodontie</a> '
            "(FFO) as part of the Dental Segmentator developments and the "
            '<a href="https://rhu-cosy.com/en/accueil-english/">Cure Overgrowth Syndromes</a>'
            " (COSY) RHU Project."
        )


class SlicerNNUNetWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)
        widget = Widget()
        self.logic = widget.logic
        self.layout.addWidget(widget)


class SlicerNNUNetTest(ScriptedLoadableModuleTest):
    """
    Runs every test except for integration test cases.
    """

    def printAndDisplay(self, msg, isWarning):
        import logging
        print(msg) if not isWarning else logging.warning(msg)
        self.delayDisplay(msg)

    def runTest(self):
        from pathlib import Path

        try:
            from SlicerPythonTestRunnerLib import RunnerLogic, RunSettings
        except ImportError:
            self.printAndDisplay("SlicerPythonTestRunner not found. Test skipped.", isWarning=True)
            return

        currentDirTest = Path(__file__).parent.joinpath("Testing")
        results = RunnerLogic().runAndWaitFinished(
            currentDirTest,
            RunSettings(extraPytestArgs=RunSettings.pytestFileFilterArgs("*TestCase.py") + ["-m not slow"])
        )

        if results.failuresNumber:
            self.printAndDisplay(f"Tests failed.\n{results.getFailingCasesString()}", isWarning=True)
        else:
            self.printAndDisplay(f"Tests OK. {results.getSummaryString()}", isWarning=False)
