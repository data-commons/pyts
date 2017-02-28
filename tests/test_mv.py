from pyts.mv import MovingAverage
import numpy as np
import numpy.testing as npt
import unittest

class MVTest(unittest.TestCase):
	
	def getATol(self,expected_arr):
		return (sum(expected_arr)/len(expected_arr))/20

	def test_should_predect_the_next_two_valus_of_ma(self):
		actual = MovingAverage(np.array([1,2,3,4,5,6,7,8]),3,4).predections
		expected = np.array([7,7.3333,7.4444,7.2592])
		absTol  = self.getATol(expected)
		npt.assert_allclose(actual,expected,rtol = 0, atol = absTol)
