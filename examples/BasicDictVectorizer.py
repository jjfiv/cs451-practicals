from typing import Dict, List, Any
import numpy as np

## USE sklearn's DictVectorizer instead!
class BasicDictVectorizer:
    def __init__(self):
        self.feature_names_ = []

    def fit_transform(self, xs: List[Dict[str, Any]]) -> np.ndarray:
        self.fit(xs)  # count up all the features, figure out column numbers
        return self.transform(xs)  # use the column numbers to convert to a matrix

    def fit(self, xs: List[Dict[str, Any]]):
        # collect feature_names and kinds
        feature_names = set([])
        for row in xs:
            for key in row.keys():
                feature_names.add(key)
        # collect columns to use:
        self.feature_names_ = sorted(feature_names)

    def transform(self, xs: List[Dict[str, Any]]) -> np.ndarray:
        D = len(self.feature_names_)
        X = np.zeros((len(xs), D))
        name_to_index = dict((name, i) for (i, name) in enumerate(self.feature_names_))

        # for each example:
        for n, row in enumerate(xs):
            # for each feature:
            for fname, fval in row.items():
                key = fname
                val = float(fval)
                index = name_to_index.get(key, -1)
                # if we haven't seen it before
                if index < 0:
                    continue
                X[n, index] = val
        return X


numberer = BasicDictVectorizer()

train_data = [
    {"x": 3, "y": 7},
    {"z": 1, "cat": True},
    {"z": "2"},
    {"z": 3},
]

test_data = [
    {"x": 1, "y": 2, "z": 3},
    {"x": 7},
    {"z": 3},
]

numberer.fit(train_data)
print("Column Names:", numberer.feature_names_)
print("=== TRAIN ===")
print(numberer.transform(train_data))
print("=== TEST ===")
print(numberer.transform(test_data))