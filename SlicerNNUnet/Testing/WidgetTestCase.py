import unittest
from pathlib import Path
from unittest.mock import MagicMock

import slicer
from packaging.version import parse

from SlicerNNUNetLib import Widget, Signal
from Testing.Utils import load_test_CT_volume


class MockSegmentationLogic:
    def __init__(self):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")
        self.setParameter = MagicMock()
        self.startSegmentation = MagicMock()
        self.stopSegmentation = MagicMock()
        self.waitForSegmentationFinished = MagicMock()
        self.loadSegmentation = MagicMock()
        self.loadSegmentation.side_effect = self.load_segmentation

    @staticmethod
    def load_segmentation():
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")


class MockInstallLogic:
    def __init__(self):
        self.progressInfo = Signal()
        self.needsRestart = False
        self.setupPythonRequirements = MagicMock(return_value=True)
        self.getInstalledNNUnetVersion = MagicMock(return_value=parse("2.2.3"))


class WidgetTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.segmentationLogic = MockSegmentationLogic()
        self.installLogic = MockInstallLogic()
        self.widget = Widget(
            segmentationLogic=self.segmentationLogic,
            installLogic=self.installLogic,
            doShowInfoWindows=False
        )
        self.node = load_test_CT_volume()
        self.widget.ui.inputSelector.setCurrentNode(self.node)
        self.widget.show()

    def assertButtonsEnabled(self):
        self.assertTrue(self.widget.ui.inputSelector.isEnabled())
        self.assertTrue(self.widget.ui.installButton.isEnabled())
        self.assertTrue(self.widget.ui.applyButton.isVisible())
        self.assertFalse(self.widget.ui.stopButton.isVisible())

    def assertButtonsDisabled(self):
        self.assertFalse(self.widget.ui.inputSelector.isEnabled())
        self.assertFalse(self.widget.ui.installButton.isEnabled())
        self.assertFalse(self.widget.ui.applyButton.isVisible())
        self.assertTrue(self.widget.ui.stopButton.isVisible())

    def test_widget_calls_install_and_segmentation_on_apply(self):
        self.installLogic.needsRestart = False

        self.widget.ui.applyButton.click()

        self.installLogic.setupPythonRequirements.assert_called_once()
        self.segmentationLogic.setParameter.assert_called_once()
        self.segmentationLogic.startSegmentation.assert_called_once_with(self.node)

        self.assertButtonsDisabled()
        self.segmentationLogic.inferenceFinished()
        self.segmentationLogic.loadSegmentation.assert_called_once()
        self.assertButtonsEnabled()

        # Check segmentation is loaded and contains the input volume name on load
        segmentations = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))
        self.assertEqual(len(segmentations), 1)
        self.assertIn(self.node.GetName(), segmentations[0].GetName())

    def test_widget_doesnt_call_segmentation_if_needs_restart_after_install(self):
        self.installLogic.needsRestart = True

        self.widget.ui.applyButton.click()
        self.widget.ui.applyButton.click()

        self.segmentationLogic.startSegmentation.assert_not_called()
        self.assertButtonsEnabled()

    def test_if_fails_during_install_doesnt_call_segmentation(self):
        self.installLogic.setupPythonRequirements.return_value = False
        self.installLogic.needsRestart = False

        self.widget.ui.applyButton.click()
        self.segmentationLogic.startSegmentation.assert_not_called()
        self.assertButtonsEnabled()

    def test_nnunet_path_is_forwarded_to_segmentation_logic(self):
        self.widget.ui.nnUNetModelPathEdit.setCurrentPath("MODEL_PATH")
        self.widget.ui.applyButton.click()
        self.segmentationLogic.setParameter.assert_called_once()
        self.assertEqual(self.segmentationLogic.setParameter.call_args[0][0].modelPath, Path("MODEL_PATH"))
