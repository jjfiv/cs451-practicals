"""
In this lab, we'll go ahead and use the sklearn API to learn a decision tree over some actual data!

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

We'll need to install sklearn.
Either use the GUI, or use pip:

    pip install scikit-learn
    # or: use install everything from the requirements file.
    pip install -r requirements.txt
"""

# We won't be able to get past these import statments if you don't install the library!
from sklearn.tree import DecisionTreeClassifier

import json  # standard python
from shared import dataset_local_path, TODO  # helper functions I made

#%% load up the data
examples = []
feature_names = set([])

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # make a big list of all the features we have:
        for name in keep.keys():
            feature_names.add(name)
        # whether or not it's poetry is our label.
        keep["y"] = info["poetry"]
        # hold onto this single dictionary.
        examples.append(keep)

#%% Convert data to 'matrices'
# NOTE: there are better ways to do this, built-in to scikit-learn. We will see them soon.

# turn the set of 'string' feature names into a list (so we can use their order as matrix columns!)
feature_order = sorted(feature_names)

# Set up our ML problem:
train_y = []
train_X = []

# Put every other point in a 'held-out' set for testing...
test_y = []
test_X = []

for i, row in enumerate(examples):
    # grab 'y' and treat it as our label.
    example_y = row["y"]
    # create a 'row' of our X matrix:
    example_x = []
    for feature_name in feature_order:
        example_x.append(float(row[feature_name]))

    # put every fourth page into the test set:
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

#%% Now actually train the model...

# Create a regression-tree object:
f = DecisionTreeClassifier(
    splitter="best",
    max_features=None,
    criterion="gini",
    max_depth=None,
    random_state=13,
)  # type:ignore

# train the tree!
f.fit(train_X, train_y)

# did it memorize OK?
print("Score on Training: {:.3f}".format(f.score(train_X, train_y)))
print("Score on Testing: {:.3f}".format(f.score(test_X, test_y)))

## Actual 'practical' assignment.
TODO(
    "1. Figure out what all of the parameters I listed for the DecisionTreeClassifier do."
)
# Consult the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
TODO("2. Pick one parameter, vary it, and find some version of the 'best' setting.")
# Default performance:
# There are 2079 training examples and 693 testing examples.
# Score on Training: 1.000
# Score on Testing: 0.889
TODO("3. Leave clear code for running your experiment!")