import unittest

import slicer
from packaging.requirements import Requirement


class InstallLogicTestCase(unittest.TestCase):
    def test_pip_check_nnunetv2_does_not_raise(self):
        # Should not raise, regardless of whether nnunetv2 is installed
        slicer.util.pip_check(Requirement("nnunetv2"))
