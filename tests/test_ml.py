import pytest
import pandas as pd
import unittest.mock
from sklearn.datasets import load_iris
import numpy as np

from src.models import MachineAlgorithm, LinReg, DTRegression

data = load_iris()["data"]


def test_preprocessing(X=data[:, :-1], y=data[:, -1]):
    prep_data = MachineAlgorithm(X, y).prepare_data()
    assert len(prep_data) == 3
    assert len(prep_data["X_train"]) > 0
    assert len(prep_data["y_train"]) > 0
    assert prep_data["X_train"].shape[1] == 3


def test_linreg(X=data[:, :-1], y=data[:, -1]):
    linreg = LinReg(X, y)
    model = linreg.prepare_model_and_data()
    y_pred = model[0].predict(model[1].transform([[1, 1, 1]]))

    assert model is not None
    assert len(y_pred) > 0

def test_kmeans(X=data[:, :-1], y=data[:, -1]):
    kmeans = DTRegression(X, y)
    model = kmeans.prepare_model_and_data()
    y_pred = model[0].predict(model[1].transform([[1, 1, 1]]))

    assert model is not None
    assert len(y_pred) > 0
