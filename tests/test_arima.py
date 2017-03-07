import pandas as pd
from datetime import date

from auto_arima import AutoARIMA
from data import long_series_with_small_values
from test_case import TestCase


class ARIMATest(TestCase):
    def test_should_be_able_to_built_a_simple_arima_model(self):
        date_range = pd.date_range(start=date(2015, 1, 1), end=date(2017, 2, 8))
        test_data = pd.Series(long_series_with_small_values(), index=date_range)
        arima_model = AutoARIMA(test_data).fit()
        actual = arima_model.predict(start='2017-02-09', end='2017-02-21', typ='levels').values
        expected = [1225.74530208, 1284.90211441, 1307.10633069, 1314.13500535, 1316.85497267,
                    1318.35159262, 1319.50087433, 1320.55153822, 1321.57420211, 1322.58891611, 1323.60137293,
                    1324.6131889, 1325.62482291]
        self.npt.assert_allclose(actual, expected, 0.01)
