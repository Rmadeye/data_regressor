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
            values = input("Enter values for prediction (sep by comma): ")
            if values == "KONIEC":
                break
            values = [int(x) for x in values.split(",")]
            y_pred = ac.select_model(model_choice, selected_var, values)
            # breakpoint()
            print(f"Predicted value: {y_pred}")

        # data = load_iris()["data"]
        # db = DBCreator("iris", data.shape[1], "a,b,c,d")

        # print("Select column to be used as independent feature (y)")
        # col_vals = db.show_data().columns.values
        # print(col_vals)
        # y_choice = int(input(f"{[i for i in range(len(col_vals))]}: "))
        # columns_to_keep = [i for i in range(data.shape[1]) if i != y_choice]
        # y = data[:, y_choice]
        # X = data[:, columns_to_keep]

        # model_choice = input("Select model: lr or lr: ")
        # if model_choice == "lr":
        #     lr = LinReg(X, y).prepare_model()
        #     vals = input(
        #         f"Enter values for prediction (column names:) {db.show_data().iloc[:,[x for x in range(0,db.show_data().shape[1]) if x != y_choice]].columns.values},  separated by comma: "
        #     )
        #     vals = np.array([int(x) for x in vals.split(",")]).reshape(1, -1)
        #     print(f"Predicted value: {lr[0].predict(vals)[0]}")
