from data import long_series
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
        self.npt.assert_allclose(exponential_smoothing.aic(), -0.02, atol=0.1)

    def test_should_choose_the_alpha_value_automatically_based_on_the_aic_values(self):
        series = pd.Series(data=[13, 17, 19, 23, 24])
        exponential_smoothing = ExponentialSmoothing(series)
        exponential_smoothing.fit()
        self.npt.assert_allclose(exponential_smoothing.aic(), -115.56, atol=0.1)

    def test_should_choose_the_alpha_value_automatically_on_large_data(self):
        series = pd.Series(long_series())
        exponential_smoothing = ExponentialSmoothing(series)
        exponential_smoothing.fit()
        self.npt.assert_allclose(exponential_smoothing.aic(), -8506.96, atol=0.1)
