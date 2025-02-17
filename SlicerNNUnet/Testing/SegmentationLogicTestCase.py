import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import qt
import slicer

from SlicerNNUNetLib import SegmentationLogic, Signal
from SlicerNNUNetLib.Parameter import Parameter
from .Utils import NNUNetTestCase, load_test_CT_volume


class MockProcess:
    def __init__(self):
        self.errorOccurred = Signal()
        self.finished = Signal()
        self.readInfo = Signal()
        self.start = MagicMock()
        self.stop = MagicMock()
        self.waitForFinished = MagicMock()


class SegmentationLogicTestCase(NNUNetTestCase):
    def setUp(self):
        super().setUp()
        self._clearScene()
        self._tmp_dir = qt.QTemporaryDir()

        self.mockError = MagicMock()
        self.mockInfo = MagicMock()

        self.process = MockProcess()
        self.logic = SegmentationLogic(self.process)
        self.logic.setParameter(Parameter(modelPath=self.get_tmp_dataset_folder()))
        self.logic.errorOccurred.connect(print)
        self.logic.errorOccurred.connect(self.mockError)
        self.logic.progressInfo.connect(print)
        self.logic.progressInfo.connect(self.mockInfo)

        self.volume = load_test_CT_volume()

    def get_tmp_dataset_folder(self) -> Path:
        dataset_folder = Path(self._tmp_dir.path()) / "Dataset111_453CT" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        dataset_folder.mkdir(parents=True, exist_ok=True)
        return dataset_folder

    def create_dataset_file(self, file_ending=None):
        dataset_dict = {
            "channel_names": {
                "0": "channel name"
            },
            "labels": {
                "background": 0,
                "Test Label 1": 1,
                "Test Label 2": 2,
                "Test Label 3": 3,
                "Test Label 4": 4,
                "Test Label 5": 5,
            },
            "numTraining": 42,
            "file_ending": file_ending or ".nii.gz"
        }

        dataset_path = self.get_tmp_dataset_folder() / "dataset.json"

        with open(dataset_path, "w") as f:
            f.write(json.dumps(dataset_dict, indent=4))

        return dataset_path

    def create_fake_segmentation(self, file_ending=None):
        import numpy as np
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        volumeId = shNode.GetItemByDataNode(self.volume)
        clonedId = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, volumeId)
        clonedVolume = shNode.GetItemDataNode(clonedId)
        array = slicer.util.arrayFromVolume(clonedVolume)
        array = np.random.randint(0, 6, array.shape, dtype=np.int32)
        slicer.util.updateVolumeFromArray(clonedVolume, array)

        file_ending = file_ending or ".nii.gz"
        slicer.util.exportNode(clonedVolume, self.logic.nnUNetOutDir.joinpath(f"out{file_ending}"))
        slicer.mrmlScene.RemoveNode(clonedVolume)

    def create_folds_folders(self, folds=(0, 1, 2, 3, 4), chkpt_name="checkpoint_final.pth"):
        for fold in folds:
            fold_path = self.get_tmp_dataset_folder() / f"fold_{fold}"
            fold_path.mkdir()
            chkpt_file = fold_path.joinpath(chkpt_name)
            with open(chkpt_file, "w"):
                pass

    def create_fake_model_dir(self, file_ending=None):
        self.create_dataset_file(file_ending)
        self.create_folds_folders()

    def test_setups_temporary_volume_for_nn_unet_runner(self):
        mockInferenceFinished = MagicMock()

        self.create_fake_model_dir()
        self.logic.inferenceFinished.connect(mockInferenceFinished)
        self.logic.startSegmentation(self.volume)

        self.process.start.assert_called_once()

        self.assertTrue(self.logic.nnUNetInDir.exists())
        self.assertTrue(self.logic.nnUNetOutDir.exists())
        self.assertEqual(len(list(self.logic.nnUNetInDir.rglob("*.nii.gz"))), 1)

        self.process.finished()
        mockInferenceFinished.assert_called_once()

    def test_exported_volume_file_ending_changes_depending_on_dataset_file(self):
        self.create_fake_model_dir(file_ending=".nrrd")
        self.logic.inferenceFinished.connect(MagicMock())
        self.logic.startSegmentation(self.volume)
        self.assertTrue(self.logic.nnUNetInDir.exists())
        self.assertEqual(len(list(self.logic.nnUNetInDir.rglob("*.nrrd"))), 1)
        self.process.finished()

    def test_segmentation_forwards_process_information(self):
        self.process.readInfo("READ")
        self.process.errorOccurred("ERROR")

        self.mockInfo.assert_called_once_with("READ")
        self.mockError.assert_called_once_with("ERROR")

    def test_informs_error_occurred_if_invalid_model_path(self):
        self.logic.startSegmentation(self.volume)

        self.mockError.assert_called_once()
        self.mockError.reset_mock()

        self.create_folds_folders()
        self.logic.startSegmentation(self.volume)
        self.mockError.assert_called_once()
        self.process.start.assert_not_called()

    def test_informs_error_occurred_if_fold_is_outside_created_ones(self):
        self.create_fake_model_dir()
        self.logic.setParameter(Parameter(modelPath=self.get_tmp_dataset_folder(), folds="5"))
        self.logic.startSegmentation(self.volume)

        self.mockError.assert_called_once()
        self.process.start.assert_not_called()

    def test_informs_error_occurred_if_dataset_name_doesnt_start_with_dataset(self):
        self.create_fake_model_dir()
        model_path = os.path.join(self._tmp_dir.path(), "StringWithoutDatasetPrefix")
        os.rename(os.path.join(self._tmp_dir.path(), "Dataset111_453CT"), model_path)

        self.logic.setParameter(Parameter(modelPath=Path(model_path)))
        self.logic.startSegmentation(self.volume)

        self.mockError.assert_called_once()
        self.process.start.assert_not_called()

    def test_loads_segmentation_names_from_data_set(self):
        self.create_fake_model_dir()
        self.logic.startSegmentation(self.volume)
        self.create_fake_segmentation()
        self.logic.loadSegmentation()

        segmentations = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))
        self.assertEqual(len(segmentations), 1)

        segmentation = segmentations[0].GetSegmentation()
        s4 = segmentation.GetNthSegmentID(4)
        self.assertIn("Test Label", segmentation.GetSegment(s4).GetName())

    def test_loads_segmentations_based_on_dataset_file_ending(self):
        self.create_fake_model_dir(file_ending=".nrrd")
        self.logic.startSegmentation(self.volume)
        self.create_fake_segmentation(file_ending=".nrrd")
        self.logic.loadSegmentation()
        self.assertEqual(self.mockError.call_count, 0)

    def test_parameters_can_be_stored_to_and_from_settings(self):
        param = Parameter(
            folds="0,1,2",
            device="cpu",
            stepSize=0.42,
            disableTta=False,
            nProcessSegmentationExport=2,
            nProcessPreprocessing=4,
            checkPointName="custom",
            modelPath=Path("PATH")
        )

        param.toSettings()
        p2 = Parameter.fromSettings()
        self.assertEqual(param, p2)
