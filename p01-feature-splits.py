# Decision Trees: Feature Splits

#%%
# Python typing introduced in 3.5: https://docs.python.org/3/library/typing.html
from typing import List

# As of Python 3.7, this exists! https://www.python.org/dev/peps/pep-0557/
from dataclasses import dataclass

# My python file (very limited for now, but we will build up shared functions)
from shared import TODO

#%%
# Let's define a really simple class with two fields:
@dataclass
class DataPoint:
    temperature: float
    frozen: bool

    def secret_answer(self) -> bool:
        return self.temperature <= 32

    def clone(self) -> "DataPoint":
        return DataPoint(self.temperature, self.frozen)


# Fahrenheit, sorry.
data = [
    # vermont temperatures; frozen=True
    DataPoint(0, True),
    DataPoint(-2, True),
    DataPoint(10, True),
    DataPoint(11, True),
    DataPoint(6, True),
    DataPoint(28, True),
    DataPoint(31, True),
    # warm temperatures; frozen=False
    DataPoint(33, False),
    DataPoint(45, False),
    DataPoint(76, False),
    DataPoint(60, False),
    DataPoint(34, False),
    DataPoint(98.6, False),
]


def is_water_frozen(temperature: float) -> bool:
    """
    This is how we **should** implement it.
    """
    return temperature <= 32


# Make sure the data I invented is actually correct...
for d in data:
    assert d.frozen == is_water_frozen(d.temperature)


def find_candidate_splits(data: List[DataPoint]) -> List[float]:
    midpoints = []
    TODO("find the midpoints!")
    return midpoints


def gini_impurity(points: List[DataPoint]) -> float:
    """
    The standard version of gini impurity sums over the classes:
    """
    p_ice = sum(1 for x in points if x.frozen) / len(points)
    p_water = 1.0 - p_ice
    return p_ice * (1 - p_ice) + p_water * (1 - p_water)
    # for binary gini-impurity (just two classes) we can simplify, because 1 - p_ice == p_water, etc.
    # p_ice * p_water + p_water * p_ice
    # 2 * p_ice * p_water
    # not really a huge difference.


def impurity_of_split(points: List[DataPoint], split: float) -> float:
    smaller = []
    bigger = []

    TODO("split the points based on the candidate split value")

    return gini_impurity(smaller) + gini_impurity(bigger)


if __name__ == "__main__":
    print("Initial Impurity: ", gini_impurity(data))
    print("Impurity of first-six (all True): ", gini_impurity(data[:6]))
    print("")
    for split in find_candidate_splits(data):
        score = impurity_of_split(data, split)
        print("splitting at {} gives us impurity {}".format(split, score))
        if score == 0.0:
            break
