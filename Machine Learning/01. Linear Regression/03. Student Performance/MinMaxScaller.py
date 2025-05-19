import numpy as np

class MinMaxScaller:
    def __init__(self):
        self.MinMax = {}
    def scaller(self,col):
        min = col.min()
        max = col.max()
        values = [(val - min)/(max - min) for val in col]
        return values, [min, max]

    def descaller(self,col,min_max):
        min, max = min_max
        descalled = [((val*(max-min))+min) for val in col]
        return descalled
    
    def MinMaxScaller(self,dataset):
        scalledDataset = np.zeros_like(dataset,dtype=float)
        for col in range(dataset.shape[1]):
            scalledDataset[:,col], min_max = self.scaller(dataset[:,col])
            self.MinMax[col] = min_max
        return scalledDataset

    def MinMaxDescaller(self,dataset):
        descalledDataset = np.zeros_like(dataset,dtype=float)
        for col in range(dataset.shape[1]):
            descalledDataset[:,col] = self.descaller(dataset[:,col],self.MinMax[col])
        return descalledDataset
