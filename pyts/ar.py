from statsmodels.tsa.ar_model import AR 

class AutoRegression(object):
	def __init__(self,data_set):
		self.data_set = data_set
		self.model = AR(self.data_set)
		self.model_fit = self.model.fit()

	def predict(self,n_ahead):
		length = len(self.data_set)
		predictions = self.model_fit.predict(start=length,end=length+n_ahead-1,dynamic=True)
		return predictions