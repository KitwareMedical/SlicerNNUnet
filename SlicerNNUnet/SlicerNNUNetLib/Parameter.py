import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Annotated, List, Tuple, Optional, Dict

import qt
from slicer.parameterNodeWrapper import parameterPack, Choice, WithinRange


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
    device: Annotated[str, Choice(["cuda", "cpu", "mps"])] = "cuda"
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

        isValid, reason = self.isValid()
        if not isValid:
            raise RuntimeError(f"Invalid nnUNet configuration. {reason}")

        args = [
            "-i", inDir.as_posix(),
            "-o", outDir.as_posix(),
            "-d", self._datasetName,
            "-tr", self._configurationNameParts[0],
            "-p", self._configurationNameParts[1],
            "-c", self._configurationNameParts[-1],
            "-f", *[str(f) for f in self._foldsAsList()],
            "-npp", self.nProcessPreprocessing,
            "-nps", self.nProcessSegmentationExport,
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
            except TypeError:
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
    def _getFirstFolderWithDatasetFile(path: Path) -> Optional[Path]:
        try:
            return next(path.rglob("dataset.json")) if path else None
        except StopIteration:
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
