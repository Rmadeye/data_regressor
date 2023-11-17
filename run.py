import numpy as np

from src.dbcreator import DBCreator
from src.models import LinReg, DTRegression
from src.ui import DataPredictor

from sklearn.datasets import load_iris

def collect_user_dataset_choice() -> str:
    print("Do you want to work on preset dataset?")
    ans1 = input("Y/n: ")
    if ans1.upper() == "Y":
        return "preset"
    elif ans1.upper() == "N":
        return "user"
    else:
        print("Wrong choice, try again")
        return collect_user_dataset_choice()
    
def collect_model_choice() -> str:
    print("Enter algorithm (lm/dt): ")
    ans = input()
    if ans.lower() == "lm" or ans.lower() == "dt":
        return ans
    else:
        print("Wrong choice, try again")
        return collect_model_choice()

def collect_user_dataset() -> DBCreator:
    print("Enter database name: ")
    db_name = input()
    print("Enter number of features: ")
    feature_number = input()
    print("Enter feature names separated by comma: ")
    feature_names = input()
    mc = DataPredictor(db_name, feature_number, feature_names)
    feature_values = ""
    while True:
        feature_values = input("Enter your features separated by comma or KONIEC to exit: ")
        if feature_values == "KONIEC":
            break
        feature_values = feature_values.split(",")
        mc.add_entry_from_input(feature_values)
    return mc


if __name__ == "__main__":
    choice = collect_user_dataset_choice()
    model_choice = collect_model_choice()
    data_predictor = DataPredictor()
    if choice == "preset":
        print(f"Columns: {data_predictor.db.return_dataframe().columns.values}")
        selected_var = int(input("Enter index of column to be used as independent feature: "))
        values = ""
        while True:
            values = input("Enter values for prediction (sep by comma) or KONIEC to exit: ")
            if values == "KONIEC":
                break
            values = [int(x) for x in values.split(",")]
            model = data_predictor.prepare_model(model_choice, selected_var)
            y_pred = data_predictor.predict(model, values)
            print(f"Predicted value: {y_pred}")

    elif choice == "user":
        custom_dataset = collect_user_dataset()
        model_choice = input("Enter algorithm (lm/dt): ")
        print(f"Columns: {custom_dataset.show_data().columns.values}")
        selected_var = int(input("Enter index of column to be used as independent feature: "))
        X, y = custom_dataset.split_dataset(selected_var)
        values = ""
        while True:
            values = input("Enter values for prediction (sep by comma) or KONIEC to exit: ")
            if values == "KONIEC":
                break
            values = np.array([int(x) for x in values.split(",")])
            # Check if the input has the correct number of features
            if len(values) != X.shape[1]:
                print(f"Error: expected {X.shape[1]} features, but got {len(values)}")
                continue
            y_pred = model_choice[0].predict(values.reshape(1, -1))
            print(f"Predicted value: {y_pred}")
        
