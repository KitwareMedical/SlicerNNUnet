import logging
from typing import Optional, Protocol

import slicer
from packaging.requirements import Requirement
from packaging.version import Version


class InstallLogicProtocol(Protocol):
    """
    Interface definition for the InstallLogic.
    Only the methods defined in this interface are stable.
    """
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

    Usage example :
    >>> logic = InstallLogic()
    >>> logic.getInstalledNNUnetVersion()
    None
    >>> logic.setupPythonRequirements()
    >>> logic.getInstalledNNUnetVersion()
    <Version('2.3.1')>
    """

    def __init__(self, doAskConfirmation=True):
        self.doAskConfirmation = doAskConfirmation
        self.needsRestart = False

    @staticmethod
    def _log(text):
        logging.info(text)

    def setupPythonRequirements(self, nnUNetRequirements: str = "nnunetv2") -> bool:
        """
        Setups 3D Slicer's Python environment with the requested nnunetv2 requirement.
        Install will proceed with the best PyTorch version for environment.

        Setup may require 3D Slicer to be restarted to fully proceed.
        """
        try:
            req = Requirement(nnUNetRequirements)
            if slicer.util.pip_check(req):
                self._log(
                    f"nnUNet is already installed ({self.getInstalledNNUnetVersion()}) "
                    f"and compatible with requested version ({nnUNetRequirements})."
                )
                return True

            self.installPyTorchExtensionAndRestartIfNeeded()
            if self.needsRestart:
                self._log("Slicer needs to be restarted before continuing install.")
                return True

            if self.doAskConfirmation:
                if not slicer.util.confirmOkCancelDisplay(
                    "nnUNet will be installed to 3D Slicer. "
                    "This install can take a few minutes. "
                    "Would you like to proceed?",
                    "nnUNet about to be installed",
                ):
                    raise RuntimeError("Install process was manually canceled by user.")

            self._log(f"Start nnUNet install with requirements : {nnUNetRequirements}")
            torchRequirement = self._installNNUnet(nnUNetRequirements)
            self._installPyTorch(torchRequirement)
            self._log("nnUNet installation completed successfully.")
            return True
        except Exception as e:
            self._log(f"Error occurred during install : {e}")
            return False

    def getInstalledNNUnetVersion(self) -> Optional[Version]:
        from importlib.metadata import version, PackageNotFoundError
        from packaging.version import parse
        try:
            return parse(version("nnunetv2"))
        except PackageNotFoundError:
            return None

    def _installNNUnet(self, nnunetRequirement: str) -> str:
        """
        Installs nnUNet while not installing SimpleITK, torch and requests.

        Slicer's SimpleITK uses a special IO class, which should not be replaced.
        Torch requires special install using SlicerPyTorch.
        Requests would require restart which is unnecessary.
        """
        if slicer.util.pip_check(Requirement("nnunetv2")):
            slicer.util.pip_uninstall("nnunetv2")

        skipped = slicer.util.pip_install(
            nnunetRequirement,
            skip_packages=["SimpleITK", "torch", "requests"],
        )

        return next(req for req in skipped if "torch" in req.lower())

    def _installPyTorch(self, torchRequirements: str) -> None:
        torchLogic = self._getTorchLogic()
        torchReq = Requirement(torchRequirements)

        if slicer.util.pip_check(torchReq):
            return

        self._log("PyTorch Python package is required. Installing... (it may take several minutes)")
        if torchLogic.installTorch(
                askConfirmation=False,
                torchVersionRequirement=str(torchReq.specifier)
        ) is None:
            raise RuntimeError(
                "Failed to correctly install PyTorch. PyTorch extension needs to be installed to use this module."
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

            import qt
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
