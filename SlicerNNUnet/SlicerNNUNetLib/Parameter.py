import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Tuple, Optional, Dict

import qt
from slicer.parameterNodeWrapper import parameterPack, Choice, WithinRange


@lru_cache(maxsize=None)
def _isNvidiaDriverInstalled() -> bool:
    """
    Checks whether the NVIDIA driver, which is required for CUDA, is installed by loading its library.
    Does not require PyTorch.
    """
    import ctypes
    import sys
    libraryNames = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
    for libraryName in libraryNames:
        try:
            ctypes.CDLL(libraryName)
            return True
        except OSError:
            continue
    return False


def availableDevices() -> List[str]:
    """
    Inference devices that can be used on this platform, the preferred one first.
    Determined without PyTorch, as it may not be installed yet when this is called:
    CUDA is not available on macOS, and MPS (Apple GPU) is only available on Apple Silicon.
    On other platforms CUDA is preferred if the NVIDIA driver is installed. CUDA is offered
    even if the driver is not detected, so that it can still be selected manually.
    """
    import platform
    import sys
    if sys.platform == "darwin":
        return ["mps", "cpu"] if platform.machine() == "arm64" else ["cpu"]
    return ["cuda", "cpu"] if _isNvidiaDriverInstalled() else ["cpu", "cuda"]


@parameterPack
@dataclass
class Parameter:
    """
    Parameters storing the NNUNet config in Python formats.
    Sets the default values as used by the Segmentation Logic.
    Parameters are compatible with 3D Slicer parameterNodeWrapper.

    Provides method to convert to nnUNet process arg list.
    """
    folds: str = ""
    device: Annotated[str, Choice(availableDevices())] = availableDevices()[0]
    stepSize: Annotated[float, WithinRange(0., 1.0)] = 0.5
    disableTta: bool = True
    nProcessPreprocessing: Annotated[int, WithinRange(1, 999)] = 1
    nProcessSegmentationExport: Annotated[int, WithinRange(1, 999)] = 1
    checkPointName: str = ""
    modelPath: Path = Path()

    def asDict(self) -> Dict:
        return asdict(self)

    def asJSon(self, indent=None):
        return json.dumps(self.asDict(), cls=_PathEncoder, indent=indent)

    def debugString(self):
        return self.asJSon(indent=4)

    def asArgList(self, inDir: Path, outDir: Path) -> List:
        import torch
        import sys

        isValid, reason = self.isValid()
        if not isValid:
            raise RuntimeError(f"Invalid nnUNet configuration. {reason}")

        npp = self.nProcessPreprocessing
        nps = self.nProcessSegmentationExport

        # Force sequential on macOS
        if sys.platform == "darwin":
            # nnU-net hangs on macOS at the end of inference (when saving the results)
            # due to multiprocessing. Disable multiprocessing to prevent the hang.
            npp = 0
            nps = 0

        args = [
            "-i", inDir.as_posix(),
            "-o", outDir.as_posix(),
            "-d", self._datasetName,
            "-tr", self._configurationNameParts[0],
            "-p", self._configurationNameParts[1],
            "-c", self._configurationNameParts[-1],
            "-f", *[str(f) for f in self._foldsAsList()],
            "-npp", npp,
            "-nps", nps,
            "-step_size", self.stepSize,
            "-device", self._getDevice(),
            "-chk", self._getCheckpointName()
        ]

        if self.disableTta:
            args.append("--disable_tta")

        return args

    def isSelectedDeviceAvailable(self) -> bool:
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return False
        if self.device == "mps" and not torch.backends.mps.is_available():
            return False
        return True

    def _getDevice(self):
        """
        Get compatible nnUNet device for current torch and hardware install
        """
        if not self.isSelectedDeviceAvailable():
            return "cpu"
        return self.device

    def toSettings(self, settings: Optional[qt.QSettings] = None, key: str = "") -> None:
        """
        Saves the current Parameters to QSettings.
        If settings is not provided, saves to the default .ini file.
        """
        key = key or self._defaultSettingsKey()
        settings = settings or qt.QSettings()
        settings.setValue(key, self.asJSon())
        settings.sync()

    @classmethod
    def fromSettings(cls, settings: Optional[qt.QSettings] = None, key: str = ""):
        """
        Creates Parameters from the saved settings.
        If settings is not provided, loads from the default .ini file.
        If Parameter is not found or contains partial data, loads available and defaults the rest.
        """
        key = key or cls._defaultSettingsKey()
        settings = settings or qt.QSettings()
        val = settings.value(key, "")
        val_dict = json.loads(val, object_hook=_PathEncoder.decodePath) if val else {}

        instance = cls()
        for k, v in val_dict.items():
            try:
                instance.setValue(k, v)
            except (TypeError, ValueError):
                # ValueError: value is not valid anymore, for example a saved device that is not available on this platform
                continue
        return instance

    @classmethod
    def _defaultSettingsKey(cls):
        return "SlicerNNUNet/Parameter"

    def _getCheckpointName(self):
        return self.checkPointName or "checkpoint_final.pth"

    def isValid(self) -> Tuple[bool, str]:
        """
        Checks if the current configuration is valid.
        Returns True and empty string if that's the case.
        False and reason for failure otherwise.
        """
        exp_struct_msg = (
            "Your model weight folder path should look like the following :\n"
            "Dataset<dataset_id>/<trainer_name>__<plan_name>__<conf_name>\n\n"
            "It should also contain a dataset.json file and fold_<i_fold> folders with model weights.\n\n"
            f"Provided model dir :\n{self.modelPath}"
        )

        if not self._isDatasetPathValid():
            return False, f"dataset.json file is missing.\n{exp_struct_msg}"

        # Check input configuration folds matches the input model folder
        missing_folds = self._getMissingFolds()
        if missing_folds:
            return False, f"Model folder is missing the following folds : {missing_folds}.\n{exp_struct_msg}"

        # Check folds with invalid weights
        folds_with_invalid_weights = self._getFoldsWithInvalidWeights()
        if folds_with_invalid_weights:
            return (
                False,
                f"Following model folds don't contain {self._getCheckpointName()} weights : "
                f"{folds_with_invalid_weights}.\n{exp_struct_msg}"
            )

        # Check configuration folder name is valid

        if not len(self._configurationNameParts) == 3:
            return (
                False,
                f"Invalid nnUNet configuration folder : {self._configurationFolder.name}\n{exp_struct_msg}"
            )

        # Check dataset folder name is valid
        if not self._isDatasetNameValid():
            return (
                False,
                f"Invalid Dataset folder name : {self._datasetName}\n{exp_struct_msg}"
            )

        return True, ""

    @staticmethod
    def _isConvertibleToInt(dataSetFolderName):
        try:
            int(dataSetFolderName)
            return True
        except ValueError:
            return False

    def _isDatasetNameValid(self):
        return self._datasetName.startswith("Dataset") or self._isConvertibleToInt(self._datasetName)

    def readSegmentIdsAndLabelsFromDatasetFile(self) -> Optional[List[Tuple[str, str]]]:
        """
        Load SegmentIds / labels pairs from the dataset file.
        """
        if not self._isDatasetPathValid():
            return None

        with open(self._datasetFilePath, "r") as f:
            dataset_dict = json.loads(f.read())
            labels = dataset_dict.get("labels")
            return [(f"Segment_{v}", k) for k, v in labels.items()]

    def readFileEndingFromDatasetFile(self) -> str:
        default = ".nii.gz"
        if not self._isDatasetPathValid():
            return default

        with open(self._datasetFilePath, "r") as f:
            dataset_dict = json.loads(f.read())
            return dataset_dict.get("file_ending", default)

    def readChannelNamesFromFile(self) -> List[str]:
        if not self._isDatasetPathValid():
            return []

        with open(self._datasetFilePath, "r") as f:
            dataset_dict = json.loads(f.read())
            channel_names = dataset_dict.get("channel_names", {})
            if isinstance(channel_names, dict):
                # Sort keys numerically when possible to keep channel order stable
                return [channel_names[k] for k in
                        sorted(channel_names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
            return channel_names if isinstance(channel_names, list) else []

    @property
    def _datasetFilePath(self) -> Optional[Path]:
        path = self._getFirstFolderWithDatasetFile(self.modelPath)
        if path is not None:
            return path

        isProvidedPathFoldDir = list(self.modelPath.glob(self._getCheckpointName()))
        if isProvidedPathFoldDir:
            return self._getFirstFolderWithDatasetFile(self.modelPath.parent)
        return None

    @staticmethod
    def _getFirstFolderWithDatasetFile(path: Path, searchDepth: int = 3) -> Optional[Path]:
        pattern = "dataset.json"
        for _ in range(searchDepth):
            dataset_path = next(path.glob(pattern), None)
            if dataset_path:
                return dataset_path
            pattern = f"*/{pattern}"
        return None

    def _foldsAsList(self) -> List[int]:
        return [int(f) for f in self.folds.strip().split(",")] if self.folds else [0]

    def _getFoldPaths(self) -> List[Tuple[int, Path]]:
        return [(fold, self._configurationFolder.joinpath(f"fold_{fold}")) for fold in self._foldsAsList()]

    def _getMissingFolds(self) -> List[int]:
        return [fold for fold, path in self._getFoldPaths() if not path.exists()]

    def _getFoldsWithInvalidWeights(self) -> List[int]:
        return [fold for fold, path in self._getFoldPaths() if not path.joinpath(self._getCheckpointName()).exists()]

    @property
    def _configurationFolder(self) -> Path:
        return self._datasetFilePath.parent

    @property
    def _datasetFolder(self) -> Path:
        return self._configurationFolder.parent

    @property
    def _datasetName(self) -> str:
        return self._datasetFolder.name

    @property
    def modelFolder(self) -> Path:
        return self._datasetFolder.parent

    @property
    def _configurationNameParts(self) -> List[str]:
        return self._configurationFolder.name.split("__")

    def _isDatasetPathValid(self) -> bool:
        return self._datasetFilePath is not None


class _PathEncoder(json.JSONEncoder):
    """
    Helper encoder to save / restore modelPath from QSettings.
    """

    def default(self, obj):
        if isinstance(obj, Path):
            return {'_path': str(obj)}
        return super().default(obj)

    @staticmethod
    def decodePath(obj):
        if '_path' in obj:
            return Path(obj['_path'])
        return obj
