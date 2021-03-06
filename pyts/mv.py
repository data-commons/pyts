import numpy as np


class MovingAverage(object):
    def __init__(self,data_set,q,num):
        self.data_set = data_set
        self.q = q
        self.num = num
        self.res = []
        self.predections = self.getMA()

    def getMA(self):
        if(self.num >= 1):
            data = self.data_set[-self.q:]
            avg = float(sum(data))/float(len(data))
            self.res = np.append(self.res,avg)
            self.data_set =np.append(self.data_set,avg)
            self.num -= 1
            return self.getMA()
        else:
            return self.res
    

