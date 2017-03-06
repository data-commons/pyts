# coding=utf-8
import math


class ExponentialSmoothing(object):
    """docstring for ExponentialSmoothing"""
    """    st=αxt+(1−α)st−1 """

    def __init__(self, time_series, alpha=0.0):
        self.time_series = time_series
        self.alpha = alpha
        self.alpha_inv = 1 - self.alpha

    def predict(self, n=1):
        fitted = self.__compute__fit(self.time_series)
        if n is 1:
            return fitted[-n:]
        for ahead in xrange(n):
            x = fitted[-1]
            m = x * self.alpha + self.alpha_inv * fitted[-1]
            fitted.append(m)
        return fitted[-n:]

    def __compute__fit(self, time_series, fitted=None):
        for _, x in enumerate(time_series):
            if fitted is None:
                fitted = [x * self.alpha]
            else:
                m = x * self.alpha + self.alpha_inv * fitted[-1]
                fitted.append(m)
        return fitted

    def aic(self):
        fitted = self.__compute__fit(self.time_series)
        n = len(self.time_series)
        k = 2
        residuals = self.time_series - fitted
        rss = sum(residuals * residuals)
        likelihood = rss / n
        return n * math.log(likelihood) + 2 * k
