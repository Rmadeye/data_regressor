import sys
import os

import pytest
import pandas as pd
import unittest.mock

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_directory)

from src.dbcreator import DBCreator


@pytest.fixture
def db_creator():
    db_name = "testname"
    feature_number = 2
    feature_names = "a,b"
    db = DBCreator(db_name, feature_number, feature_names)
    return db


# Test to create the DBCreator instance and capture values
def test_creating_db(db_creator):
    assert db_creator.db_name == "testname"
    assert db_creator.feature_number == 2
    assert db_creator.feature_names == ["a", "b"]
    assert all(db_creator.df.columns == pd.DataFrame(columns=["a", "b"]).columns)


@pytest.mark.parametrize("features", [[1, 2]])
def test_add_entry(db_creator, features: list):
    db_creator._add_entry(features)

    assert db_creator.show_data().shape[0] == 1, "Incorrect data shape"


def test_input_add(db_creator):
    with unittest.mock.patch("builtins.input", return_value="1,2"):
        db_creator.get_data_from_input()

        assert db_creator.show_data().shape[0] == 1, "Incorrect data shape"
