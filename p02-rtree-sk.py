"""
In this lab, we'll go ahead and use the sklearn API to learn a regression tree over some REAL data!

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

We'll need to install sklearn.
Either use the GUI, or use pip:

    pip install scikit-learn
    # or: use install everything from the requirements file.
    pip install -r requirements.txt
"""

# We won't be able to get past these import statments if you don't install the library!
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import math

import csv  # standard python
from shared import dataset_local_path  # helper function I made

# These are the numeric columns in the dataset:
NUMERIC_COLUMNS = [
    "X",
    "Y",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain",
]
LABEL_COLUMN = "area"

examples = []

with open(dataset_local_path("forest-fires.csv")) as fp:
    rows = csv.reader(fp)
    # first row of file is a 'header':
    header = next(rows)
    # for each row, take the current values, and zip them with the header into a dictionary:
    for row in rows:
        as_dict = dict(zip(header, row))
        examples.append(as_dict)

# Set up our ML problem:
train_y = []
train_X = []

# Put every other point in a 'held-out' set for testing...
test_y = []
test_X = []

for i, row in enumerate(examples):
    example_x = []
    for n in NUMERIC_COLUMNS:
        example_x.append(float(row[n]))
    example_y = math.log(1 + float(row[LABEL_COLUMN]))

    if i % 4 == 0:
        test_X.append(example_x)
        test_y.append(example_y)
    else:
        train_X.append(example_x)
        train_y.append(example_y)


print(
    "There are {} training examples and {} testing examples.".format(
        len(train_y), len(test_y)
    )
)

# Create a regression-tree object:
f = RandomForestRegressor(max_depth=1)

# train the tree!
f.fit(train_X, train_y)

# did it memorize OK?
print("Score on Training: {}".format(f.score(train_X, train_y)))
print("Score on Testing: {}".format(f.score(test_X, test_y)))
