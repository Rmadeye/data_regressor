from typing import List, Union
import os

import pandas as pd
import numpy as np


class DBCreator:
    def __init__(self, db_name: str, feature_number: str, feature_names: Union[str,list]):
        self.db_name = db_name
        self.feature_number = int(feature_number)
        self.feature_names = feature_names
        if isinstance(self.feature_names,str):
            self.feature_names = feature_names.split(",")
        assert self.db_name, "Database must have a name"
        assert self.feature_number >= 2, "Feature number too small"
        assert len(self.feature_names) == self.feature_number

        self.df = pd.DataFrame(columns=self.feature_names)

    def _add_entry(self, features: List):
        self.df = pd.concat(
            [self.df, pd.DataFrame([features], columns=self.feature_names)],
            ignore_index=True,
        )

    def add_entry_from_input(self, features: Union[List, str] = None):
        features = features.split(",") if isinstance(features, str) else features
        self._add_entry(features)

    def return_dataframe(self):
        return self.df
