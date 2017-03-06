# coding=utf-8
import math
import numpy as np


class ExponentialSmoothing(object):
    """docstring for ExponentialSmoothing"""
    """    st=αxt+(1−α)st−1 """

    def __init__(self, time_series, alpha=0.0):
        self.time_series = time_series
        self.alpha = alpha

    def predict(self, n=1):
        fitted = self.__compute__fit(self.time_series)
        if n is 1:
            return fitted[-n:]
        for ahead in xrange(n):
            x = fitted[-1]
            m = x * self.alpha + self.alpha_inv() * fitted[-1]
            fitted.append(m)
        return fitted[-n:]

    def __compute__fit(self, time_series, fitted=None):
        for _, x in enumerate(time_series):
            if fitted is None:
                fitted = [x * self.alpha]
            else:
                m = x * self.alpha + self.alpha_inv() * fitted[-1]
                fitted.append(m)
        return fitted

    def fit(self):
        max_aic = np.inf
        alpha = 0.0
        for x in np.linspace(0.1, 0.999999):
            self.alpha = x
            self.__compute__fit(self.time_series)
            aic = self.aic()
            if aic < max_aic:
                max_aic = aic
                alpha = x
        self.alpha = alpha

    def alpha_inv(self):
        return 1 - self.alpha

    def aic(self):
        fitted = self.__compute__fit(self.time_series)
        n = len(self.time_series)
        k = 2
        residuals = self.time_series - fitted
        rss = sum(residuals * residuals)
        likelihood = rss / n
        return n * math.log(likelihood) + 2 * k
