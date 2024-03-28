import importlib.metadata
import importlib.util
import logging
from importlib.metadata import version, PackageNotFoundError
from subprocess import CalledProcessError
from typing import Optional, Union, Protocol

import qt
import slicer
from packaging.requirements import Requirement
from packaging.version import parse, Version

from .Signal import Signal


class InstallLogicProtocol(Protocol):
    """
    Interface definition for the InstallLogic.
    Only the methods defined in this interface are stable.
    """
    progressInfo: Signal
    needsRestart: bool

    def setupPythonRequirements(self, nnUNetRequirements: str) -> None:
        pass

    def getInstalledNNUnetVersion(self) -> Optional[Version]:
        pass


class InstallLogic:
    r"""
    Class responsible for installing nnUNet in a clean way for 3D Slicer usage.
    Makes sure that SimpleITK and requests packages are not overwritten during install.
    Makes sure that torch is installed separately by PyTorch module.

    Copied and adapted from :
    https://github.com/lassoan/SlicerTotalSegmentator/blob/main/TotalSegmentator/TotalSegmentator.py

    Usage example :
    >>> logic = InstallLogic()
    >>> logic.progressInfo.connect(print)
    >>> logic.getInstalledNNUnetVersion()
    None
    >>> logic.setupPythonRequirements()
    >>> logic.getInstalledNNUnetVersion()
    <Version('2.3.1')>
    """

    def __init__(self, doAskConfirmation=True):
        self.progressInfo = Signal("str")
        self.doAskConfirmation = doAskConfirmation
        self.needsRestart = False

    def _log(self, text):
        logging.info(text)
        self.progressInfo(text)

    def setupPythonRequirements(self, nnUNetRequirements: str = "nnunetv2") -> bool:
        """
        Setups 3D Slicer's Python environment with the requested nnunetv2 requirement.
        Install will proceed with the best PyTorch version for environment.

        Setup may require 3D Slicer to be restarted to fully proceed.
        """
        try:
            if self.isPackageInstalledAndCompatible(nnUNetRequirements):
                self._log("nnUNet is already installed and compatible with requested version.")
                return True

            self.installPyTorchExtensionAndRestartIfNeeded()
            if self.needsRestart:
                self._log("Slicer needs to be restarted before continuing install.")
                return True

            self._log(f"Start nnUNet install with requirements : {nnUNetRequirements}")
            self._installPandas()
            self._downgradePillowToLessThan10_1()
            torchRequirement = self._installNNUnet(nnUNetRequirements)
            self._installPyTorch(torchRequirement)
            self._downgradeDynamicNetworkArchitecture()
            self._log("nnUNet installation completed successfully.")
            return True
        except Exception as e:
            self._log(f"Error occurred during install : {e}")
            return False

    def getInstalledNNUnetVersion(self) -> Version:
        return self.getInstalledPackageVersion("nnunetv2")

    @classmethod
    def isPackageInstalledAndCompatible(cls, req: Union[str, Requirement]) -> bool:
        return cls.isPackageInstalled(req) and cls.isInstalledPackageCompatible(req)

    @classmethod
    def isPackageInstalled(cls, req: Union[str, Requirement]) -> bool:
        return cls.getInstalledPackageVersion(req) is not None

    @classmethod
    def isInstalledPackageCompatible(cls, req: Union[str, Requirement]) -> bool:
        """
        Checks if the requirement is installed in this environment and if the installed version is compatible with req.
        """
        req = cls.asRequirement(req)
        installedVersion = cls.getInstalledPackageVersion(req)
        return installedVersion in req.specifier if installedVersion is not None else True

    @classmethod
    def needsToInstallRequirement(cls, req: Union[str, Requirement]):
        """
        Check if the input requirement matches the current environment. Otherwise, return False.
        """
        req = cls.asRequirement(req)
        if req.marker is not None and not req.marker.evaluate():
            return False

        # Check if the package is installed and compatible with requirement.
        return not cls.isPackageInstalled(req) or not cls.isInstalledPackageCompatible(req)

    @classmethod
    def getInstalledPackageVersion(cls, req: Union[str, Requirement]) -> Optional[Version]:
        req = cls.asRequirement(req)
        try:
            return parse(version(req.name))
        except PackageNotFoundError:
            return None

    @classmethod
    def asRequirement(cls, req: Union[str, Requirement]) -> Requirement:
        """
        Converts input string to Requirement instance.
        """
        return req if isinstance(req, Requirement) else Requirement(req)

    def _installNNUnet(self, nnunetRequirement: str) -> str:
        """
        Installs nnUNet while not installing SimpleITK, torch and requests.

        Slicer's SimpleITK uses a special IO class, which should not be replaced.
        Torch requires special install using SlicerPyTorch.
        Requests would require restart which is unnecessary.
        """
        nnUNetPackagesToSkip = [
            'SimpleITK',
            'torch',
            'requests',
        ]

        # Install nnunetv2 with selected dependencies only
        self._uninstallNNUnetIfNeeded()
        skipped = self.pipInstallSelective('nnunetv2', nnunetRequirement, nnUNetPackagesToSkip)
        torchRequirement = next(req for req in skipped if "torch" in req.lower())
        return torchRequirement

    def _uninstallNNUnetIfNeeded(self):
        if not self.isPackageInstalled(Requirement("nnunetv2")):
            return
        self.pip_uninstall("nnunetv2")

    def _downgradeDynamicNetworkArchitecture(self) -> None:
        """
        Workaround: fix incompatibility of dynamic_network_architectures==0.4 with totalsegmentator==2.0.5.
        Revert to the last working version: dynamic_network_architectures==0.2
        """
        if parse(version("dynamic_network_architectures")) == parse("0.4"):
            self._log(
                f'dynamic_network_architectures package version is incompatible. Installing working version...')
            self.pip_install("dynamic_network_architectures==0.2.0")

    def _installPyTorch(self, torchRequirements: str) -> None:
        torchLogic = self._getTorchLogic()
        torchRequirements = Requirement(torchRequirements)

        if self.isPackageInstalled(torchRequirements) and self.isInstalledPackageCompatible(torchRequirements):
            return

        if not self.isInstalledPackageCompatible(torchRequirements):
            if self.doAskConfirmation:
                self._requestPermissionForTorchInstallOrRaise(torchRequirements)

        self._log("PyTorch Python package is required. Installing... (it may take several minutes)")
        if torchLogic.installTorch(
                askConfirmation=self.doAskConfirmation,
                torchVersionRequirement=str(torchRequirements.specifier)
        ) is None:
            raise RuntimeError(
                "Failed to correctly install PyTorch. PyTorch extension needs to be installed to use this module."
            )

    @classmethod
    def _requestPermissionForTorchInstallOrRaise(cls, torchRequirements: Requirement) -> None:
        """
        Request user permission to install PyTorch. Raises RuntimeError if canceled by user.
        """
        torchVersion = cls.getInstalledPackageVersion(torchRequirements)
        reqVersion = torchRequirements.specifier

        msg = (
            f"PyTorch version {torchVersion} is not compatible with this module."
            f" Version required is {reqVersion}."
        )

        ret = qt.QMessageBox.question(
            None,
            "Invalid Torch version detection",
            msg + " Would you like to upgrade your PyTorch version?"
        )

        if ret == qt.QMessageBox.No:
            raise RuntimeError(
                msg +
                f' You can use "PyTorch Util" module to install PyTorch'
                f' with version requirement set to: {reqVersion}'
            )

    def installPyTorchExtensionAndRestartIfNeeded(self):
        """
        Install PytorchUtils if not installed and raises RuntimeError if canceled by user or install was unsuccessful.
        """
        try:
            self._getTorchLogic()
        except RuntimeError:
            if not self.doAskConfirmation:
                raise

            ret = qt.QMessageBox.question(
                None,
                "Pytorch extension not found.",
                "This module requires PyTorch extension. Would you like to install it?\n\n"
                "Slicer will need to be restarted before continuing the install."
            )
            if ret == qt.QMessageBox.No:
                raise

            self.needsRestart = True
            self.installTorchUtils()

    @staticmethod
    def installTorchUtils() -> None:
        """
        Installs PytorchUtils from server and raises RuntimeError if install was unsuccessful.
        """
        extensionManager = slicer.app.extensionsManagerModel()
        extName = "PyTorch"
        if extensionManager.isExtensionInstalled(extName):
            return

        if not extensionManager.installExtensionFromServer(extName):
            raise RuntimeError(
                "Failed to install PyTorch extension from the servers. "
                "Manually install to continue."
            )

    @classmethod
    def _getTorchLogic(cls) -> "PyTorchUtilsLogic":
        """
        Returns torch utils logic if available. Otherwise, raise RuntimeError.
        """
        try:
            import PyTorchUtils  # noqa
            return PyTorchUtils.PyTorchUtilsLogic()
        except ModuleNotFoundError:
            raise RuntimeError(
                "This module requires PyTorch extension. "
                "Install it from the Extensions Manager and restart Slicer to continue."
            )

    def _downgradePillowToLessThan10_1(self) -> None:
        """
        Pillow version that is installed in Slicer (10.1.0) is too new,
        it is incompatible with several nnUNet dependencies.
        Attempt to uninstall and install an older version before any of the packages import  it.
        """
        needToInstallPillow = parse(version("pillow")) >= parse("10.1")
        if needToInstallPillow:
            self.pip_install("pillow<10.1")

    def _installPandas(self) -> None:
        """
        Installs pandas if necessary.
        """
        try:
            import pandas
        except ModuleNotFoundError:
            self.pip_install("pandas")

    def pipInstallSelective(self, packageToInstall, installCommand, packagesToSkip):
        """
        Installs a Python package, skipping a list of packages.
        Return the list of skipped requirements (package name with version requirement).
        """
        installCommand = self.cleanPyPiRequirement(installCommand)
        self.pip_install(f"{installCommand} --no-deps")

        # Install all dependencies but the ones listed in packagesToSkip
        requirements = importlib.metadata.requires(packageToInstall)
        if not requirements:
            return []

        # Update meta file to remove packages to skip.
        # Necessary to avoid having skipped dependencies installed into 3D Slicer during pip updates.
        self._removeSkippedPackagesFromMetaDataFile(packageToInstall, packagesToSkip)

        skippedRequirements = []
        for requirement in requirements:
            skipThisPackage = False
            for packageToSkip in packagesToSkip:
                if requirement.startswith(packageToSkip):
                    # Do not install
                    skipThisPackage = True
                    break

            if skipThisPackage:
                skippedRequirements.append(requirement)
            elif self.needsToInstallRequirement(requirement):
                # Install sub dependencies and make sure they enforce requirements not to install
                self.pipInstallSelective(Requirement(requirement).name, requirement, packagesToSkip)

        return skippedRequirements

    @classmethod
    def _removeSkippedPackagesFromMetaDataFile(cls, packageToInstall, packagesToSkip):
        def doSkipLine(metaLine):
            if not metaLine.startswith("Requires-Dist: "):
                return False
            for packageToSkip in packagesToSkip:
                if packageToSkip in metaLine:
                    return True
            return False

        with open(cls.packageMetaFilePath(packageToInstall), "r+") as file:
            filteredLines = "".join([line for line in file if not doSkipLine(line)])
            file.seek(0)
            file.write(filteredLines)
            file.truncate()

    @staticmethod
    def packageMetaFilePath(packageToInstall):
        import importlib.metadata
        return [p for p in importlib.metadata.files(packageToInstall) if 'METADATA' in str(p)][0].locate()

    @staticmethod
    def cleanPyPiRequirement(requirement) -> str:
        """
        Returns requirement string compatible with Slicer pip_install call.
        """
        import re
        req = Requirement(requirement)

        # Get any extra pypi to install from the requirements extras
        extras = [extra for extra in req.extras]
        extras = str(extras) if extras else ""

        # Handle special case where extra would be in the marker instead of the extra spec
        # Takes into account the ruff ; extra == 'dev' -> ruff[dev] case
        extra_pattern = "extra == "
        req_marker = str(req.marker)
        if not extras and req_marker.startswith(extra_pattern):
            req_marker = re.sub(r"\W+", '', req_marker.replace(extra_pattern, ""))
            extras = f"[{req_marker}]"

        return f"{req.name}{extras}{req.specifier}"

    def pip_install(self, package) -> None:
        """
        Install and log install of input package.
        """
        self._log(f'- Installing {package}...')
        try:
            slicer.util.pip_install(package)
        except CalledProcessError as e:
            self._log(f"Install returned non-zero exit status : {e}. Attempting to continue...")

    def pip_uninstall(self, package) -> None:
        """
        Uninstall and log uninstall of input package.
        """
        self._log(f'- Uninstall {package}...')
        try:
            slicer.util.pip_uninstall(package)
        except CalledProcessError as e:
            self._log(f"Uninstall returned non-zero exit status : {e}. Attempting to continue...")
