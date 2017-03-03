import pandas as pd
import numpy as np

class Seasonality(object):
	def __init__(self,trainData,numDays):
		self.trainData = trainData
		self.numDays = numDays
		self.prediciton = self.predectSeasonality(self.trainData,self.numDays)

	def addDays(self,train,numDays,init):
		pred=[]
		for i in range(init+1, init+numDays+1):
			if(i < len(train)):
				pred.append(train[i])
			else:
				pred.append(pred[i-len(train)-len(pred)])
		return pred

	def findSeason(self,trainData):
		data = trainData
		data.index = range(len(data))
		minVal = np.min(data)
		args = np.where(data == minVal)[0]
		return(args[2]-args[1])


	def addDates(self,predected, lastDate):
		dates = pd.date_range(lastDate, periods = len(predected)+1)[1:]
		pred = pd.Series(predected)
		pred.index = dates
		return pred

	def predectSeasonality(self,trainData,numDays):
		index = trainData.index
		lastDate = index[-1]
		season = self.findSeason(trainData)
		numTrainObser = season*2
		train = trainData[-numTrainObser:]
		train.index = range(numTrainObser)
		lastVal = train[numTrainObser-1]
		for i in range(0,numTrainObser):
			if(lastVal == train[i]):
				predected = self.addDays(train,numDays,i)
				break
		predectedWithDates = self.addDates(predected,lastDate)
		return predectedWithDates



