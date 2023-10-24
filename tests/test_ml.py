import sys
import os

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_directory)

import pytest
import pandas as pd
import unittest.mock
from sklearn.datasets import load_iris
import numpy as np

from src.models import MachineAlgorithm, LinReg

data = load_iris()["data"]


def test_preprocessing(X=data[:, :-1], y=data[:, -1]):
    prep_data = MachineAlgorithm(X, y).prepare_data()

    assert len(prep_data) == 5
    assert len(prep_data["X_train"]) > 0
    assert len(prep_data["X_test"]) > 0
    assert len(prep_data["y_train"]) > 0
    assert len(prep_data["y_test"]) > 0
    assert prep_data["X_train"].shape[1] == 3
    assert prep_data["X_test"].shape[1] == 3


def test_linreg(X=data[:, :-1], y=data[:, -1]):
    linreg = LinReg(X, y)
    model, _, X_test, y_test = linreg.prepare_model()
    y_pred = model.predict(X_test)

    assert model is not None
    assert len(y_pred) > 0
    assert len(y_pred) == len(linreg.prepare_data()["y_test"])
    mse = np.mean((y_pred - y_test) ** 2)
    assert np.isclose(mse, 0.05, atol=0.01)
