import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from src.models import LinReg, DTRegression
from src.dbcreator import DBCreator


class AutomaticChosen:
    dataset: np.array = load_iris()["data"]
    dbname: str = "iris"
    cols: list = load_iris()["feature_names"]

    def __init__(self):
        self.db = DBCreator(self.dbname, self.dataset.shape[1], self.cols)

    def split_dataset(self, selected_var: int):
        columns_to_keep = [i for i in range(self.dataset.shape[1]) if i != selected_var]

        return self.dataset[:, columns_to_keep], self.dataset[:, selected_var]

    def select_model(
        self,
        chosen_model: str = "lm",
        selected_var: int = -1,
        values_to_predict: list = [0, 0, 0],
    ):
        X, y = self.split_dataset(selected_var)
        if chosen_model == "lm":
            lr = LinReg(X, y).prepare_model()
            vals = np.array(values_to_predict).reshape(-1, 1).reshape(1, -1)
            assert type(vals) == np.ndarray
            vals = lr[1].transform(vals)
            output = lr[0].predict(vals)
            return output[0]
        if chosen_model == 'dt':
            dt = DTRegression(X, y).prepare_model()
            vals = np.array(values_to_predict).reshape(-1, 1).reshape(1, -1)
            assert type(vals) == np.ndarray
            vals = dt[1].transform(vals)
            output = dt[0].predict(vals)
            return output[0]
        

class ManualChosen(DBCreator):

    def __init__(self, db_name, feature_number, feature_names) -> None:
        super().__init__(db_name, feature_number, feature_names)

    def split_dataset(self, selected_var: int):
        columns_to_keep = [i for i in range(self.df.shape[1]) if i != selected_var]

        return self.df.iloc[:, columns_to_keep], self.df.iloc[:, selected_var]
    
    def select_model(
        self,
        chosen_model: str = "lm",
        selected_var: int = -1,
        values_to_predict: list = [0, 0, 0],
    ):
        X, y = self.split_dataset(selected_var)
        if chosen_model == "lm":
            lr = LinReg(X, y).prepare_model()
            vals = np.array(values_to_predict).reshape(-1, 1).reshape(1, -1)
            assert type(vals) == np.ndarray
            vals = lr[1].transform(vals)
            output = lr[0].predict(vals)
            return output[0]
        else:
            km = DTRegression(X, y).prepare_model()
            vals = np.array(values_to_predict).reshape(-1, 1).reshape(1, -1)
            assert type(vals) == np.ndarray
            vals = km[1].transform(vals)
            output = km[0].predict(vals)
            return output[0]
