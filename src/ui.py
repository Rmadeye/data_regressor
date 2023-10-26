import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from src.models import LinReg
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
        

class ManualChosen:
    db_name: str = ""
    feature_number: int
    feature_names: list

    def __init__(self) -> None:
        self.db_name = input("Enter database name: ")
        self.feature_number = int(input("Enter number of features: "))
        self.feature_names = input("Enter feature names separated by comma: ")
        self.feature_names = self.feature_names.split(",")
        assert self.db_name, "Database must have a name"
        assert self.feature_number >= 2, "Feature number too small"
        assert len(self.feature_names) == self.feature_number

        self.df = pd.DataFrame(columns=self.feature_names)

    def _add_entry(self, features: list):
        self.df = pd.concat(
            [self.df, pd.DataFrame([features], columns=self.feature_names)],
            ignore_index=True,
        )

    def get_data_from_input(self):
        features = input("Enter your features values separated by comma")
        features = features.split(",")
        self._add_entry(features)


    def split_dataset(self, selected_var: int):
        columns_to_keep = [i for i in range(self.df.shape[1]) if i != selected_var]

        return self.df[:, columns_to_keep], self.df[:, selected_var]
    
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