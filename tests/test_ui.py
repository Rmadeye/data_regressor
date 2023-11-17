import sys
import os

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_directory)

import pytest
import pandas as pd
import unittest.mock
from sklearn.datasets import load_iris
import numpy as np
import pytest

from src.ui import DataPredictor, ManualDatasetChosen
from src.models import *

@pytest.fixture
def dataset():
    return np.random.rand(200, 3)


@pytest.mark.parametrize("selected_var", [2])
def test_loading_preset(selected_var: int):
    ac = DataPredictor()
    compare_ds = load_iris()["data"]
    assert ac.dataset.shape == compare_ds.shape
    X_user, y_user = ac.split_dataset(selected_var)
    cols_to_keep = [i for i in range(compare_ds.shape[1]) if i != selected_var]

    X_iris = compare_ds[:, cols_to_keep]
    y_iris = compare_ds[:, selected_var]

    assert all(y_iris) == all(y_user)
    assert np.all(X_iris == X_user)

@pytest.mark.parametrize(
    "chosen_model, selected_var, values_to_predict", [(["lm","km"], [3,3], [[1, 1, 1],[1,1,1]])])
def test_linreg(chosen_model, selected_var, values_to_predict):
    ac = DataPredictor()
    predict = ac.select_model(str(chosen_model), selected_var, values_to_predict)

    assert predict != 'None'

@pytest.mark.parametrize("db_name, feature_number, feature_names, selected_var", [("test", 3, ["a", "b", "c"], 1)])
def test_manual_chosen(db_name, feature_number, feature_names, dataset, selected_var):
    mc = ManualDatasetChosen(db_name, feature_number, feature_names)
    assert len(feature_names) == feature_number
    assert len(mc.df.columns) == feature_number
    assert mc.df.shape[1] == feature_number
    for i in range(len(dataset)):
        mc.add_entry_from_input(dataset[i])
    assert mc.df.shape == dataset.shape
    X_user, y_user = mc.split_dataset(selected_var)
    assert X_user.shape[0] == y_user.shape[0] > 0

    predict = mc.select_model("lm", selected_var, [1, 1])
    predict_2 = mc.select_model("km", selected_var, [1, 1])

    assert predict != 'None' and predict_2 != 'None'


    # X_output = dataset[:, cols_to_keep]
    # y_output = dataset[:, selected_var]

    # assert all(y_output) == all(y_user)
    # assert np.all(X_output == X_user)


