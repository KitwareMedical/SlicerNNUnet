from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _, translate

from SlicerNnUNetLib import Widget


class SlicerNnUNet(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SlicerNNUnet")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Thibault PELLETIER (Kitware SAS)"]
        self.parent.helpText = _(
            "This extension is meant to streamline the integration of nnUnet based models into 3D Slicer.<br>"
            "It allows for quick and relable nnUNet dependency installation in 3D Slicer environment and provides"
            " simple logic to launch nnUNet prediction on given directories.<br><br>"
            "The installation steps are based on the work done in the Slicer Total Segmentator extension"
            " (https://github.com/lassoan/SlicerTotalSegmentator)"
        )
        self.parent.acknowledgementText = _("This file was originally developed by Thibault Pelletier (Kitware SAS)")


class SlicerNnUNetWidget(ScriptedLoadableModuleWidget):
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
