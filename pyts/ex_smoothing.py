class ExponentialSmoothing(object):
	"""docstring for ExponentialSmoothing"""
	def __init__(self, dataset,alpha=0.0):
		self.dataset = dataset
		self.alpha = alpha

	def predict(self,n=1):
		smoothning_series = []
		for idx,x in enumerate(self.dataset):
			if(idx is 0):
				smoothning_series.append(x)
			else:
				m = x * self.alpha + (1.0-self.alpha) * smoothning_series[idx-1]
				smoothning_series.append(m)

		return smoothning_series[-1]
		