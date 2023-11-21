import numpy as np

from src.ui import DataPredictor
from src.collector_funcs import collect_user_dataset_choice, collect_model_choice, collect_user_dataset_data


if __name__ == "__main__":
    choice = collect_user_dataset_choice()
    model_choice = collect_model_choice()

    if choice == "preset":
        data_predictor = DataPredictor()
        print(f"Columns: {data_predictor.db.return_dataframe().columns.values}")
        selected_var = int(input("Enter index of column to be used as independent feature: "))
        values = ""
        while True:
            values = input("Enter values for prediction (sep by comma) or KONIEC to exit: ")
            if values == "KONIEC":
                break
            values = [int(x) for x in values.split(",")]
            model = data_predictor.prepare_model(model_choice, selected_var)
            y_pred = data_predictor.predict_output(model, values)
            print(f"Predicted value: {y_pred}")

    elif choice == "user":
        dataset, db_name, feature_number, feature_names = collect_user_dataset_data()
        data_predictor = DataPredictor(mode='manual', 
                            dbname=db_name,
                            cols=feature_names,
                                dataset=dataset)
        print(f"Columns: {data_predictor.db.return_dataframe().columns.values}")
        selected_var = int(input("Enter index of column to be used as independent feature: "))
        values = ""
        while True:
            values = input("Enter values for prediction (sep by comma) or KONIEC to exit: ")
            if values == "KONIEC":
                break
            values = np.array([int(x) for x in values.split(",")])
            model = data_predictor.prepare_model(model_choice, selected_var)
            y_pred = data_predictor.predict_output(model, values)
            print(f"Predicted value: {y_pred}")
