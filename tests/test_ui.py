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

from src.ui import AutomaticChosen
from src.models import *


@pytest.mark.parametrize("selected_var", [2])
def test_loading_preset(selected_var: int):
    ac = AutomaticChosen()
    compare_ds = load_iris()["data"]
    assert ac.dataset.shape == compare_ds.shape
    X_user, y_user = ac.split_dataset(selected_var)
    cols_to_keep = [i for i in range(compare_ds.shape[1]) if i != selected_var]

    X_iris = compare_ds[:, cols_to_keep]
    y_iris = compare_ds[:, selected_var]

    assert all(y_iris) == all(y_user)
    assert np.all(X_iris == X_user)


@pytest.mark.parametrize(
    "chosen_model, selected_var, values_to_predict", [("lr", 3, [1, 1, 1])]
)
def test_linreg(chosen_model, selected_var, values_to_predict):
    ac = AutomaticChosen()
    predict = ac.select_model(str(chosen_model), selected_var, values_to_predict)

    assert predict != 'None'
