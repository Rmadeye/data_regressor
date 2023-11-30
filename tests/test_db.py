import pytest
import pandas as pd
import unittest.mock

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

def test_input_add(db_creator):
    with unittest.mock.patch("builtins.input", return_value="1,1"):
        features = input().split(",")
        features = [float(f) for f in features]
        db_creator._add_entry(features)

        assert db_creator.return_dataframe().shape[0] == 1, "Incorrect data shape"
