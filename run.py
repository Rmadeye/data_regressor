from src.dbcreator import DBCreator
from src.models import *
import numpy as np
from src.ui import AutomaticChosen, ManualChosen

from sklearn.datasets import load_iris


if __name__ == "__main__":
    print("Do you want to work on preset dataset?")
    ans1 = input("Y/n: ")

    if ans1.upper() == "Y":
        ac = AutomaticChosen()
        model_choice = input("Enter algorithm (lm/dt): ")
        print(f"Columns: {ac.db.show_data().columns.values}")
        selected_var = int(input("Enter index of column to be used as independent feature: "))
        values = ""
        while True:
            values = input("Enter values for prediction (sep by comma) or KONIEC to exit: ")
            if values == "KONIEC":
                break
            values = [int(x) for x in values.split(",")]
            y_pred = ac.select_model(model_choice, selected_var, values)
            print(f"Predicted value: {y_pred}")

    elif ans1.upper() == 'N':
        print("Enter database name: ")
        db_name = input()
        print("Enter number of features: ")
        feature_number = input()
        print("Enter feature names separated by comma: ")
        feature_names = input()
        mc = ManualChosen(db_name, feature_number, feature_names)
        feature_values = ""
        while True:
            feature_values = input("Enter your features separated by comma or KONIEC to exit: ")
            if feature_values == "KONIEC":
                break
            feature_values = feature_values.split(",")
            mc.get_data_from_input(feature_values)
        model_choice = input("Enter algorithm (lm/dt): ")
        print(f"Columns: {mc.show_data().columns.values}")
        selected_var = int(input("Enter index of column to be used as independent feature: "))
        # # training
        X, y = mc.split_dataset(selected_var)
        if model_choice == "lm":
            model = LinReg(X, y).prepare_model()
        elif model_choice == "dt":
            model = DTRegression(X, y).prepare_model()
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
            y_pred = model[0].predict(values.reshape(1, -1))
            print(f"Predicted value: {y_pred}")
        
