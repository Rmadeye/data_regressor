import pytest
import unittest.mock as mock

from src.collector_funcs import collect_user_dataset_choice, collect_model_choice, collect_user_dataset_data



@mock.patch("builtins.input", side_effect=['Y', 'N','dummy'])
def test_collect_user_dataset_choice(mock_input):
    assert collect_user_dataset_choice() == "preset"
    assert collect_user_dataset_choice() == "user"

    assert mock_input.call_count == 2

@mock.patch("builtins.input", side_effect=['lm', 'dt','dummy'])
def test_collect_model_choice(mock_input):
    assert collect_model_choice() == "lm"
    assert collect_model_choice() == "dt"

    assert mock_input.call_count == 2

@mock.patch("builtins.input", side_effect=['test', '2','test1,test2,test3','1,1,1','2,2,2','3,3,3','KONIEC'])
def test_collect_user_dataset_data(mock_input):
    dataset, dbname, feature_number, feature_names = collect_user_dataset_data()
    assert dataset.shape == (3, 3)
    assert dbname == "test"
    assert feature_number == "2"
    assert feature_names == "test1,test2,test3"