from pyts.ex_smoothning import ExponentialSmoothing
import numpy as np
import numpy.testing as npt
import unittest

class ExponentialSmoothingTest(unittest.TestCase):
    def test_should_predect_the_next_valus_of_series(self):
        exponential_smoothing = ExponentialSmoothing(np.array([13,17,19,23,24]),alpha=0.9)
        expected = 23.86
        npt.assert_allclose(exponential_smoothing.predict(1),expected,0.01)