import slicer
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
            "(FFO) as part of the "
            '<a href="https://github.com/gaudot/SlicerDentalSegmentator/">Dental Segmentator</a>'
            " developments and the "
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

    def onReload(self):
        """
        Customization of reload to allow reloading of the SlicerNNUNetLib files.
        """
        import imp

        packageName = "SlicerNNUNetLib"
        submoduleNames = ["Signal", "Parameter", "InstallLogic", "SegmentationLogic", "Widget"]
        f, filename, description = imp.find_module(packageName)
        package = imp.load_module(packageName, f, filename, description)
        for submoduleName in submoduleNames:
            print(f"Reloading {packageName}.{submoduleName}")
            f, filename, description = imp.find_module(submoduleName, package.__path__)
            try:
                imp.load_module(packageName + '.' + submoduleName, f, filename, description)
            finally:
                f.close()

        ScriptedLoadableModuleWidget.onReload(self)


class SlicerNNUNetTest(ScriptedLoadableModuleTest):
    def runTest(self):
        from pathlib import Path
        from SlicerNNUNetLib import InstallLogic

        try:
            from SlicerPythonTestRunnerLib import RunnerLogic, RunSettings, isRunningInTestMode
        except ImportError:
            slicer.util.warningDisplay("Please install SlicerPythonTestRunner extension to run the self tests.")
            return

        if InstallLogic().getInstalledNNUnetVersion() is None:
            slicer.util.warningDisplay("Please install nnUNet to run the self tests of this extension.")
            return

        currentDirTest = Path(__file__).parent.joinpath("Testing")
        results = RunnerLogic().runAndWaitFinished(
            currentDirTest,
            RunSettings(extraPytestArgs=RunSettings.pytestFileFilterArgs("*TestCase.py") + ["-m not slow"]),
            doRunInSubProcess=not isRunningInTestMode()
        )

        if results.failuresNumber:
            raise AssertionError(f"Test failed: \n{results.getFailingCasesString()}")

        slicer.util.delayDisplay(f"Tests OK. {results.getSummaryString()}")
