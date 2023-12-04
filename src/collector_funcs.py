import numpy as np
from enum import Enum


class AnswerCollector(Enum):
    YES = "Y"
    NO = "N"
    LM = "LM"
    DT = "DT"
    KONIEC = "KONIEC"

    @classmethod
    def choices(cls, answer: str):
        return answer.upper()

def collect_user_dataset_choice() -> str:
    print("Do you want to work on preset dataset?")
    ans = AnswerCollector.choices(input("Y/n: "))
    if ans == AnswerCollector.YES.value:
        return "preset"
    elif ans == AnswerCollector.NO.value:
        return "user"
    else:
        print("Wrong choice, try again")
        return collect_user_dataset_choice()


    
def collect_model_choice() -> str:
    ans = AnswerCollector.choices(input("Enter algorithm (lm/dt): "))
    if ans == "LM" or ans == "DT":
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
        if AnswerCollector.choices(feature_values) == "KONIEC":
            break
        feature_values = feature_values.split(",")
        dataset.append(feature_values)
    return np.array(dataset), db_name, feature_number, feature_names