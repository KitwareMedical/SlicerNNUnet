import unittest

from SlicerNNUNetLib import InstallLogic


class InstallLogicTestCase(unittest.TestCase):
    def test_clean_pypi_requirements_rewrites_extra_into_brackets(self):
        self.assertEqual(InstallLogic.cleanPyPiRequirement("ruff ; extra == 'dev'"), "ruff[dev]")

    def test_clean_pypi_requirements_removes_spaces_from_req_string(self):
        self.assertEqual(InstallLogic.cleanPyPiRequirement("  nibabel >=2.3.0 "), "nibabel>=2.3.0")

    def test_requirements_for_python_versions_outside_slicer_are_marked_not_needed_to_install(self):
        self.assertTrue(InstallLogic.isPackageInstalled("numpy"))
        self.assertFalse(InstallLogic.isInstalledPackageCompatible("numpy < 1.0"))
        self.assertFalse(InstallLogic.needsToInstallRequirement("numpy < 1.0; python_version<'3.9'"))

    def test_requirements_for_unspecified_python_version_are_marked_needed_to_install(self):
        self.assertTrue(InstallLogic.isPackageInstalled("numpy"))
        self.assertFalse(InstallLogic.isInstalledPackageCompatible("numpy < 1.0"))
        self.assertTrue(InstallLogic.needsToInstallRequirement("numpy < 1.0"))

    def test_not_installed_package_is_marked_as_not_installed_and_not_compatible(self):
        self.assertFalse(InstallLogic.isPackageInstalledAndCompatible("not_package"))
