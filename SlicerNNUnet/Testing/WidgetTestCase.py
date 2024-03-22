import unittest

from SlicerNnUNetLib import Widget


class SlicerNNUNetInstallWidgetTestCase(unittest.TestCase):
    def test_can_be_displayed(self):
        widget = Widget()
        widget.show()
