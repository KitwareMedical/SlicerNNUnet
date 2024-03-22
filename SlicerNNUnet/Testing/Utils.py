import unittest

import slicer
from pathlib import Path


class NnUNetTestCase(unittest.TestCase):
    def setUp(self):
        self._clearScene()

    @staticmethod
    def _clearScene():
        slicer.app.processEvents()
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def tearDown(self):
        slicer.app.processEvents()


def _dataFolderPath():
    return Path(__file__).parent.joinpath("Data")


def load_test_CT_volume():
    import SampleData
    SampleData.SampleDataLogic().downloadDentalSurgery()
    return list(slicer.mrmlScene.GetNodesByName("PostDentalSurgery"))[0]
