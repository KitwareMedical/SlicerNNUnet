from unittest.mock import MagicMock

import pytest
import slicer

from SlicerNnUNetLib import SegmentationLogic
from .Utils import NnUNetTestCase, load_test_CT_volume, _dataFolderPath


class SegmentationLogicTestCase(NnUNetTestCase):
    def setUp(self):
        self._clearScene()
        self.logic = SegmentationLogic()
        self.volume = load_test_CT_volume()

    @pytest.mark.slow
    def test_can_run_segmentation(self):
        inferenceFinishedMock = MagicMock()
        errorMock = MagicMock()
        infoMock = MagicMock()

        self.logic.inferenceFinished.connect(inferenceFinishedMock)
        self.logic.errorOccurred.connect(errorMock)
        self.logic.progressInfo.connect(infoMock)

        self.logic.inferenceFinished.connect(print)
        self.logic.errorOccurred.connect(print)
        self.logic.progressInfo.connect(print)

        self.logic.setModelPath(_dataFolderPath())
        if not self.logic.isModelPathValid():
            pytest.skip("Failed to find any test nnUNet segmentation model")

        self.logic.startSegmentation(self.volume)
        while not inferenceFinishedMock.called and not errorMock.called:
            slicer.app.processEvents()

        inferenceFinishedMock.assert_called()
        infoMock.assert_called()
        errorMock.assert_not_called()

        segmentation = self.logic.loadSegmentation()
        self.assertIsNotNone(segmentation)
