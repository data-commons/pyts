from pyts.ex_smoothing import ExponentialSmoothing
import numpy as np
import pandas as pd

from test_case import TestCase


class ExponentialSmoothingTest(TestCase):
    def test_should_predict_the_next_value_of_np_array(self):
        exponential_smoothing = ExponentialSmoothing(np.array([13, 17, 19, 23, 24]), alpha=0.9)
        expected = [23.86]
        self.npt.assert_allclose(exponential_smoothing.predict(1), expected, 0.01)

    def test_should_predict_the_next_two_values_of_the_series(self):
        series = pd.Series(data=[13, 17, 19, 23, 24])
        exponential_smoothing = ExponentialSmoothing(series, alpha=0.9)
        expected = [23.86, 23.86]
        self.npt.assert_allclose(exponential_smoothing.predict(2), expected, atol=0.01)

    def test_should_calculate_aic_values_of_the_model(self):
        series = pd.Series(data=[13, 17, 19, 23, 24])
        exponential_smoothing = ExponentialSmoothing(series, alpha=0.9)
        self.npt.assert_allclose(exponential_smoothing.aic(), -8.41, atol=0.1)
