from pyts.seasonality import Seasonality

from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import numpy.testing as npt
import unittest

class TestAR(unittest.TestCase):

	def test_for_seasonality_prediciton_in_arima_model_1(self):
		'''This is basic test to check the finctionality of seasonal component of ARIMA model'''
		#Creating series of numbers 1 to 4 iteratively
		input_time_series = pd.Series(range(1,5)*3)
		#Creating Index of dates for time series
		input_time_series.index = pd.date_range(datetime.today().date(), periods = len(input_time_series))
		#Calculating actual values
		actual = Seasonality(input_time_series,3).prediciton
		#Calculating Expected values
		expected_index_start = datetime.today().date() + timedelta(days=12)
		expected = pd.Series([1,2,3])
		expected.index = pd.date_range(expected_index_start, periods = len(expected))
		npt.assert_allclose(actual,expected)

	def test_for_seasonality_prediciton_in_arima_model_2(self):
		'''This test id to check if Number of predictions grater than seasonality'''
		input_time_series = pd.Series(range(1,5)*3)
		#Creating Index of dates for time series
		input_time_series.index = pd.date_range(datetime.today().date(), periods = len(input_time_series))
		#Calculating actual values
		actual = Seasonality(input_time_series,7).prediciton
		#Calculating Expected values
		expected_index_start = datetime.today().date() + timedelta(days=12)
		expected = pd.Series([1,2,3,4,1,2,3])
		expected.index = pd.date_range(expected_index_start, periods = len(expected))
		npt.assert_allclose(actual,expected)

"""
	def test_for_finding_seasonality_1(self):
		'''Finding seasonality'''
		actual = Seasonality.findSeason(range(5)*3)
		expected = 5
		npt.assert_allclose(actual,expected)

"""
