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
        smoothing_series = self.__compute__series(self.time_series)
        if n is 1:
            return smoothing_series[-n:]
        for ahead in xrange(n):
            x = smoothing_series[-1]
            m = x * self.alpha + self.alpha_inv * smoothing_series[-1]
            smoothing_series.append(m)
        return smoothing_series[-n:]

    def __compute__series(self, time_series, smoothing_series=None):
        for _, x in enumerate(time_series):
            if smoothing_series is None:
                smoothing_series = [x]
            else:
                m = x * self.alpha + self.alpha_inv * smoothing_series[-1]
                smoothing_series.append(m)
        return smoothing_series

    def aic(self):
        prediction = self.__compute__series(self.time_series)
        n = len(self.time_series)
        k = 2
        residuals = self.time_series - prediction
        rss = sum(residuals * residuals)
        likelihood = rss / n
        return n * math.log(likelihood) + 2 * k
