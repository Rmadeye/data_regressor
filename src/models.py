from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


class MachineAlgorithm:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def prepare_data(self):
        sc = StandardScaler()
        sc.fit(self.X)
        return {
            "Scaler": sc,
            "X_train": sc.transform(self.X),
            "y_train": self.y,
        }


class LinReg(MachineAlgorithm):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.model = LinearRegression()

    def prepare_model_and_data(self):
        data = self.prepare_data()
        self.model.fit(data["X_train"], data["y_train"])

        return self.model, data['Scaler']
    
    
class DTRegression(MachineAlgorithm):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.model = KMeans()

    def prepare_model_and_data(self):
        data = self.prepare_data()
        self.model.fit(data["X_train"], data["y_train"])

        return self.model, data['Scaler']
