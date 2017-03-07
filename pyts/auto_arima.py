import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import numpy as np


class AutoARIMA(object):
    def __init__(self, time_series, order=None):
        self.time_series = time_series
        self.order = order
        seasonal, seasonal_adjusted = self.__decompose_series(time_series=self.time_series)
        self.seasonal_adjusted = seasonal_adjusted
        if order is None:
            self.order = self.__order(seasonal_adjusted)
        self.model = None

    def fit(self):
        self.model = ARIMA(self.seasonal_adjusted, self.order)
        self.model = self.model.fit(disp=0)
        return self

    def predict(self, start, end, typ):
        return self.model.predict(start=start, end=end, typ=typ)

    def __decompose_series(self, time_series):
        decomposed = sm.tsa.seasonal_decompose(time_series)
        decomposed_ts = pd.concat([decomposed.seasonal, decomposed.trend, decomposed.resid], axis=1)
        decomposed_ts.columns = ['seasonal', 'trend', 'residual']
        seasonal_adjusted = time_series - decomposed_ts['seasonal']
        return decomposed_ts['seasonal'], seasonal_adjusted

    def __order(self, series):
        # brute forcing for getting p,d,q values with minimum aic value
        (p, d, q) = (0, 0, 0)
        aic = []
        for x in range(np.argmax(acf(series) < 0.5) + 2):
            for y in [1, 2]:
                for z in range(np.argmax(pacf(series) < 0.5) + 2):
                    try:
                        tmp = ARIMA(series, order=(x, y, z)).fit(disp=0).aic
                        aic.append(tmp)
                        if min(aic) == tmp:
                            (p, d, q) = (x, y, z)
                    except:
                        pass  # Ignoring the error
        return p, d, q
