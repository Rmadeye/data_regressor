import numpy as np


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

def collect_user_dataset_data() -> tuple:
    dataset = []
    print("Enter database name: ")
    db_name = input()
    print("Enter number of features: ")
    feature_number = input()
    print("Enter feature names separated by comma: ")
    feature_names = input()
    feature_values = ""
    while True:
        feature_values = input("Enter your features separated by comma or KONIEC to exit: ")
        if feature_values == "KONIEC":
            break
        feature_values = feature_values.split(",")
        dataset.append(feature_values)
    return np.array(dataset), db_name, feature_number, feature_names