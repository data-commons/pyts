from pyts.ar import AutoRegression
import numpy as np
import numpy.testing as npt
import unittest

class TestAR(unittest.TestCase):

	def getATol(self,expected):
		return (sum(expected)/len(expected))/20

	def test_should_use_ols_for_auto_regression_and_predict_the_next_two_values_for_given_array(self):
		ar_model = AutoRegression(np.array([1,2,3,4,5,6,7,8]))
		actual = ar_model.predict(2)
		expected = np.array([9.0,10.0])
		absTol  = self.getATol(expected)
 		npt.assert_allclose(actual,expected,rtol = 0, atol = absTol)