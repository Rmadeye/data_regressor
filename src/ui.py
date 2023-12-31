import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from src.models import LinReg, DTRegression
from src.dbcreator import DBCreator


class DataPredictor:
    mode: str = "preset"
    data: np.array = load_iris() if mode == "preset" else None
    dataset: np.array = data["data"] if mode == "preset" else None
    dbname: str = "iris" if mode == "preset" else None
    cols: list = data["feature_names"] if mode == "preset" else None

    def __init__(self, mode: str = "preset", data = None,
                 dataset = None, dbname = None, cols = None):
        if mode != "preset":
            self.dbname  = dbname
            self.dataset = dataset
            self.cols = cols
        self.db = DBCreator(self.dbname, self.dataset.shape[1], self.cols)

    def split_dataset(self, selected_var: int):
        columns_to_keep = [i for i in range(self.dataset.shape[1]) if i != selected_var]

        return self.dataset[:, columns_to_keep], self.dataset[:, selected_var]

    def prepare_model(self,
                      chosen_model: str = "lm",
                      selected_var: int = -1):
        X, y = self.split_dataset(selected_var)
        model = LinReg(X, y) if chosen_model == "lm" else DTRegression(X, y)
        return model.prepare_model_and_data()
    
    def predict_output(self, model, values_to_predict: list = [0, 0, 0]):
        vals = np.array(values_to_predict).reshape(-1, 1).reshape(1, -1)
        assert type(vals) == np.ndarray
        vals = model[1].transform(vals)
        output = model[0].predict(vals)
        return output[0]