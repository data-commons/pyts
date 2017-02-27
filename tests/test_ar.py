from pyts.ar import AR
import unittest

class TestAR(unittest.TestCase):

    def test_upper(self):
        ar_model = AR([1,2,3,4,5])
        actual = ar_model.predict(2)
        expected = [6,7]
        self.assertEqual(actual,[6,7])