import pytest
import slicer

from SlicerNNUNetLib import Widget
from SlicerNNUNetLib.Parameter import Parameter
from .Utils import NNUNetTestCase, load_test_CT_volume, _dataFolderPath


@pytest.mark.slow
class IntegrationTestCase(NNUNetTestCase):
    def test_widget_can_run_nn_unet_segmentation_logic(self):
        if not Parameter(modelPath=_dataFolderPath()).isValid():
            pytest.skip(f"Skipped : no nnUNet model available in {_dataFolderPath()}")

        volume = load_test_CT_volume()
        widget = Widget(doShowInfoWindows=False)
        widget.logic.errorOccurred.connect(print)
        widget.logic.progressInfo.connect(print)
        widget.ui.inputSelector.setCurrentNode(volume)
        widget.ui.nnUNetModelPathEdit.setCurrentPath(_dataFolderPath())

        widget.ui.applyButton.click()
        widget.logic.waitForSegmentationFinished()
        slicer.app.processEvents()

        segmentations = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))
        self.assertEqual(len(segmentations), 1)

    def test_can_install_nn_unet_v2(self):
        # Uninstall previous version if necessary
        widget = Widget(doShowInfoWindows=False)
        widget.installLogic.doAskConfirmation = False

        packages_to_remove = ["torch", "nnunetv2", "pandas"]
        for package in packages_to_remove:
            slicer.util.pip_uninstall(package)

        self.assertFalse(widget.installLogic.isPackageInstalledAndCompatible("nnunetv2"))
        widget.ui.installButton.click()

        self.assertTrue(widget.installLogic.isPackageInstalledAndCompatible("nnunetv2"))

        # Check nnUNetV2 doesn't have torch as listed requirement
        with open(widget.installLogic.packageMetaFilePath("nnunetv2"), "r") as f:
            self.assertFalse("Requires-Dist: torch" in f.read())
