from sklearn.datasets import load_iris
import numpy as np
import pytest

from src.ui import DataPredictor

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
    "chosen_model, selected_var, values_to_predict", [("lm", 2, [1, 1, 1]), ("dt", 2, [1, 1, 1])])
def test_linreg(chosen_model, selected_var, values_to_predict):
    data_predictor = DataPredictor()
    model= data_predictor.prepare_model(str(chosen_model), selected_var)
    predict = data_predictor.predict_output(model=model,  values_to_predict=values_to_predict)
    assert predict != 'None'

@pytest.mark.parametrize("db_name, feature_number, feature_names, selected_var", [("test", 3, ["a", "b", "c"], 1)])
def test_manual_chosen(db_name, feature_number, feature_names, dataset, selected_var):
    mc = DataPredictor(mode='manual', dbname=db_name, cols=feature_names, dataset=dataset)
    assert len(feature_names) == feature_number
    assert mc.dataset.shape[1] == feature_number
    assert mc.dataset.shape == dataset.shape
    X_user, y_user = mc.split_dataset(selected_var)
    assert X_user.shape[0] == y_user.shape[0] > 0

    model = mc.prepare_model("lm", selected_var)
    model_2 = mc.prepare_model("dt", selected_var)
    predict = mc.predict_output(model, [1,1])
    predict_2 = mc.predict_output(model_2, [1,1])

    assert predict != 'None'
    assert predict_2 != 'None'

