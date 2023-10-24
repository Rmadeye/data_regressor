from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


class MachineAlgorithm:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def prepare_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0
        )
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        return {
            "Scaler": sc,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }


class LinReg(MachineAlgorithm):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.model = LinearRegression()

    def prepare_model(self):
        data = self.prepare_data()
        self.model.fit(data["X_train"], data["y_train"])

        return self.model, data["X_test"], data["y_test"]
