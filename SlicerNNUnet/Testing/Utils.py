import unittest
from pathlib import Path

import qt
import slicer


class NNUNetTestCase(unittest.TestCase):
    """
    Base class for every SlicerNNUNet tests.
    Clears scene before each test and saves / restores modules settings to avoid leaking into settings.
    """
    def setUp(self):
        self._clearScene()
        self._saveSettings()

    @staticmethod
    def _clearScene():
        slicer.app.processEvents()
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def tearDown(self):
        slicer.app.processEvents()
        self._restoreSettings()

    def _saveSettings(self):
        self._savedSettings = self._removeSettings()

    def _restoreSettings(self):
        self._removeSettings()
        settings = qt.QSettings()
        for k, v in self._savedSettings.items():
            settings.setValue(k, v)
        settings.sync()

    @staticmethod
    def _removeSettings():
        removed = {}
        settings = qt.QSettings()
        for k in settings.allKeys():
            if "SlicerNNUNet/" in k:
                removed[k] = settings.value(k)
                settings.remove(k)
        settings.sync()
        return removed


def _dataFolderPath():
    return Path(__file__).parent.joinpath("Data")


def load_test_CT_volume():
    import SampleData
    SampleData.SampleDataLogic().downloadDentalSurgery()
    return list(slicer.mrmlScene.GetNodesByName("PostDentalSurgery"))[0]
