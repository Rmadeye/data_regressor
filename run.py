from src.dbcreator import DBCreator
from src.models import *
import numpy as np
from src.ui import AutomaticChosen

from sklearn.datasets import load_iris


if __name__ == "__main__":
    print("Do you want to work on preset dataset?")
    ans1 = input("Y/n: ")

    if ans1.upper() == "Y":
        ac = AutomaticChosen()
        model_choice = input("Enter algorithm (lm/km): ")
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
        db = DBCreator(db_name, feature_number, feature_names)
        
