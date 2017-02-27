from pyts.mv import MovingAverage
import numpy as np
import numpy.testing as npt
import unittest

class MVTest(unittest.TestCase):
    def test_should_predect_the_next_two_valus_of_ma(self):
        actual = MovingAverage(np.array([1,2,3,4,5,6,7,8]),3,4)
        expected = np.array([7,7.3333,7.4444,7.2592])
        npt.assert_allclose(actual,expected)