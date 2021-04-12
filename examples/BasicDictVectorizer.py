from typing import Dict, List, Any
import numpy as np


def guess_kind(x: Any) -> str:
    if x == True or x == False:
        return "numeric"
    elif isinstance(x, int) or isinstance(x, float):
        return "numeric"
    else:
        # assume 'categorical', e.g., color=red
        return "categorical"


assert guess_kind(True) == "numeric"
assert guess_kind(False) == "numeric"
assert guess_kind(1.0) == "numeric"
assert guess_kind(-2) == "numeric"
assert guess_kind("blah") == "categorical"


def merge_kinds(kind1: str, kind2: str) -> str:
    if kind1 == kind2:
        return kind1
    return "categorical"


assert merge_kinds("numeric", "categorical") == "categorical"
assert merge_kinds("categorical", "categorical") == "categorical"
assert merge_kinds("numeric", "numeric") == "numeric"

## USE sklearn's DictVectorizer instead!
class BasicDictVectorizer:
    def __init__(self):
        self.feature_names_ = []
        self.categoricals = {}

    def fit_transform(self, xs: List[Dict[str, Any]]) -> np.ndarray:
        self.fit(xs)
        return self.transform(xs)

    def fit(self, xs: List[Dict[str, Any]]):
        # collect feature_names and kinds
        feature_names = {}
        for row in xs:
            for key, val in row.items():
                if feature_names.get(key, "unknown") == "categorical":
                    continue
                kind = guess_kind(val)
                if key not in feature_names:
                    feature_names[key] = kind
                else:
                    feature_names[key] = merge_kinds(feature_names[key], kind)
        # collect columns to use:
        self.feature_names_ = []
        for fname, fkind in feature_names.items():
            if fkind == "categorical":
                values = set(str(row[fname]) for row in xs if fname in row)
                self.categoricals[fname] = values
                for v in values:
                    self.feature_names_.append("{}={}".format(fname, v))
            else:
                self.feature_names_.append(fname)
        # sort!
        self.feature_names_ = sorted(self.feature_names_)

    def transform(self, xs: List[Dict[str, Any]]) -> np.ndarray:
        D = len(self.feature_names_)
        X = np.zeros((len(xs), D))
        name_to_index = dict((name, i) for (i, name) in enumerate(self.feature_names_))

        # for each example:
        for n, row in enumerate(xs):
            # for each feature:
            for fname, fval in row.items():
                key = fname
                # if it's categorical
                if fname in self.categoricals:
                    key = "{}={}".format(fname, fval)
                    val = 1
                else:
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
    {"z": 1, "cat": True, "color": "red"},
    {"color": "blue"},
    {"color": "red", "z": 3},
]

test_data = [
    {"x": 1, "y": 2, "z": 3},
    {"color": "green", "x": 7},
    {"color": "red", "z": 3},
]

numberer.fit(train_data)
print("Column Names:", numberer.feature_names_)
print("=== TRAIN ===")
print(numberer.transform(train_data))
print("=== TEST ===")
print(numberer.transform(test_data))