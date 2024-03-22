import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, List

import qt
import slicer

from .Signal import Signal


@dataclass
class NNUnetConfiguration:
    folds: List[int] = (0,)
    device: str = "cuda"
    step_size: float = 0.5
    disable_tta: bool = True


class SegmentationLogicProtocol(Protocol):
    inferenceFinished: Signal
    errorOccurred: Signal
    progressInfo: Signal

    def startSegmentation(
            self,
            volumeNode: "slicer.vtkMRMLScalarVolumeNode",
            nnUnetConf: NNUnetConfiguration
    ) -> None:
        pass

    def stopSegmentation(self):
        pass

    def waitForSegmentationFinished(self):
        pass

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        pass


class SegmentationLogic:
    def __init__(self):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")

        self.inferenceProcess = qt.QProcess()
        self.inferenceProcess.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.inferenceProcess.finished.connect(self.onFinished)
        self.inferenceProcess.errorOccurred.connect(self.onErrorOccurred)
        self.inferenceProcess.readyRead.connect(self.onCheckStandardOutput)

        self._nnUNet_predict_path = None
        self._modelPath = None
        self._tmpDir = qt.QTemporaryDir()

    @property
    def _dataSetPath(self):
        return next(self._modelPath.rglob("dataset.json"))

    def isModelPathValid(self):
        try:
            return self._dataSetPath.exists()
        except StopIteration:
            return False

    def setModelPath(self, path):
        self._modelPath = Path(path)

    def __del__(self):
        self.stopSegmentation()

    def onCheckStandardOutput(self):
        info = bytes(self.inferenceProcess.readAll().data()).decode()
        if info:
            self.progressInfo(info)

    def onErrorOccurred(self, *_):
        self.errorOccurred(bytes(self.inferenceProcess.readAllStandardError().data()).decode())

    def onFinished(self, *_):
        self.inferenceFinished()

    def startSegmentation(
            self,
            volumeNode: "slicer.vtkMRMLScalarVolumeNode",
            nnUnetConf: NNUnetConfiguration = None
    ) -> None:
        """Run the segmentation on a slicer volumeNode, get the result as a segmentationNode"""
        self._stopInferenceProcess()
        self._prepareInferenceDir(volumeNode)
        self._startInferenceProcess(nnUnetConf)

    def stopSegmentation(self):
        self._stopInferenceProcess()

    def waitForSegmentationFinished(self):
        self.inferenceProcess.waitForFinished(-1)

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        try:
            return slicer.util.loadSegmentation(self._outFile)
        except StopIteration:
            raise RuntimeError(f"Failed to load the segmentation.\nCheck the inference folder content {self._outDir}")

    def _stopInferenceProcess(self):
        if self.inferenceProcess.state() == self.inferenceProcess.Running:
            self.progressInfo("Stopping previous inference...\n")
            self.inferenceProcess.kill()

    @staticmethod
    def _nnUNetPythonDir():
        return Path(sys.executable).parent.joinpath("..", "lib", "Python")

    @classmethod
    def _findUNetPredictPath(cls):
        # nnUNet install dir depends on OS. For Windows, install will be done in the Scripts dir.
        # For Linux and MacOS, install will be done in the bin folder.
        nnUNetPaths = ["Scripts", "bin"]
        for path in nnUNetPaths:
            predict_paths = list(sorted(cls._nnUNetPythonDir().joinpath(path).glob("nnUNetv2_predict*")))
            if predict_paths:
                return predict_paths[0].resolve()

        return None

    def _startInferenceProcess(self, nnUnetConf: NNUnetConfiguration):
        """
        Run the nnU-Net V2 inference script
        """
        import torch

        # Check the nnUNet predict script is correct
        nnUnetPredictPath = self._findUNetPredictPath()
        if not nnUnetPredictPath:
            self.errorOccurred("Failed to find nnUNet predict path.")
            return

        # Check the provided model folder contains a correct dataset.json file
        if not self.isModelPathValid():
            self.errorOccurred(
                "nnUNet weights are not correctly installed."
                f" Missing path:\n{self._dataSetPath.as_posix()}"
            )
            return

        # Check input configuration folds matches the input model folder
        nnUnetConf = nnUnetConf or NNUnetConfiguration()
        configuration_folder = self._dataSetPath.parent

        missing_folds = []
        for fold in nnUnetConf.folds:
            if not configuration_folder.joinpath(f"fold_{fold}").exists():
                missing_folds.append(fold)

        if missing_folds:
            self.errorOccurred(
                f"Wrong configuration ({configuration_folder}).\n"
                f"Model folder is missing the following folds : {missing_folds}."
            )
            return

        # Check that the conf folder matches the nnUNet training pattern
        dataset_folder = configuration_folder.parent
        dataset_name = dataset_folder.name
        model_folder = dataset_folder.parent
        conf_parts = configuration_folder.name.split("__")
        if not len(conf_parts) == 3:
            self.errorOccurred(
                "Invalid nnUNet configuration folder."
                " Expected folder name such as <trainer_name>__<plan_name>__<conf_name>"
            )
            return

        # setup environment variables
        # not needed, just needs to be an existing directory
        os.environ['nnUNet_preprocessed'] = model_folder.as_posix()

        # not needed, just needs to be an existing directory
        os.environ['nnUNet_raw'] = model_folder.as_posix()
        os.environ['nnUNet_results'] = model_folder.as_posix()

        # Construct the command for the nnunnet inference script
        device = nnUnetConf.device if torch.cuda.is_available() else "cpu"
        args = [
            "-i", self._inDir.as_posix(),
            "-o", self._outDir.as_posix(),
            "-d", dataset_name,
            "-tr", conf_parts[0],
            "-p", conf_parts[1],
            "-c", conf_parts[-1],
            "-f", *[str(f) for f in nnUnetConf.folds],
            "-step_size", nnUnetConf.step_size,
            "-device", device
        ]

        if nnUnetConf.disable_tta:
            args.append("--disable_tta")

        self.progressInfo("nnUNet preprocessing...\n")
        self.inferenceProcess.start(nnUnetPredictPath, args, qt.QProcess.Unbuffered | qt.QProcess.ReadOnly)

    @property
    def _outFile(self) -> str:
        return next(file for file in self._outDir.rglob("*.nii*")).as_posix()

    def _prepareInferenceDir(self, volumeNode):
        self._tmpDir.remove()
        self._outDir.mkdir(parents=True)
        self._inDir.mkdir(parents=True)

        # Name of the volume should match expected nnUNet conventions
        self.progressInfo(f"Transferring volume to nnUNet in {self._tmpDir.path()}\n")
        volumePath = self._inDir.joinpath("volume_0000.nii.gz")
        assert slicer.util.exportNode(volumeNode, volumePath)
        assert volumePath.exists(), "Failed to export volume for segmentation."

    @property
    def _outDir(self):
        return Path(self._tmpDir.path()).joinpath("output")

    @property
    def _inDir(self):
        return Path(self._tmpDir.path()).joinpath("input")
