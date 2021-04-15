from dataclasses import dataclass
from sys import dont_write_bytecode
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from shared import bootstrap_accuracy, dataset_local_path, bootstrap_auc, simple_boxplot
import json, random


@dataclass
class LinearModel:
    weights: np.ndarray

    @staticmethod
    def random(D: int) -> "LinearModel":
        weights = np.random.randn(D + 1, 1)
        return LinearModel(weights)

    def improve_feature_weight(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        feature_id: int,
        real_change: float = 0.0001,
        measure=lambda m, X, y: m.score(X, y),
    ) -> bool:
        changes = []
        start_weights = np.copy(self.weights)
        start_score = measure(self, train_X, train_y)
        # try zero:
        self.weights[feature_id] = 0.0
        changes.append((measure(self, train_X, train_y), np.copy(self.weights)))
        for dir in [-1, +1]:  # try bigger and smaller
            for step in [
                0.001,
                0.01,
                0.1,
                1,
                2,
                4,
                8,
                16,
                32,
                64,
            ]:  # try a range of steps
                weight = start_weights[feature_id] + dir * step
                self.weights[feature_id] = weight
                now_score = measure(self, train_X, train_y)
                changes.append(
                    (now_score, np.copy(self.weights))
                )  # score accuracy/auc/whatver
        (best_score, best_weights) = max(changes, key=lambda t: t[0])
        if (best_score - start_score) >= real_change:
            self.weights = best_weights
            return True
        else:
            self.weights = start_weights
            return False

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        assert self.weights[:D].shape == (D, 1)
        # Matrix multiplication; sprinkle transpose and assert to get the shapes you want (or remember Linear Algebra)... or both!
        output = np.dot(self.weights[:D].transpose(), X.transpose())
        assert output.shape == (1, N)
        return (output + self.weights[-1]).reshape((N,))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Take whether the points are above or below our hyperplane as a prediction. """
        return self.decision_function(X) > 0

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Take predictions and compute accuracy. """
        y_hat = self.predict(X)
        return metrics.accuracy_score(np.asarray(y), y_hat)  # type:ignore

    def compute_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Distance to hyperplane is used for AUC-style metrics. """
        return metrics.roc_auc_score(y, self.decision_function(X))  # type:ignore


#%% load up the data
examples = []
ys = []

##
# Notice that we're using hand-designed features here, not text features:
##

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # whether or not it's poetry is our label.
        ys.append(info["poetry"])
        # hold onto this single dictionary.
        examples.append(keep)

## CONVERT TO MATRIX:

feature_numbering = DictVectorizer(sort=True, sparse=False)
X = feature_numbering.fit_transform(examples)

print("Features as {} matrix.".format(X.shape))


## SPLIT DATA:

RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
rX_tv, rX_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
rX_train, rX_vali, y_train, y_vali = train_test_split(
    rX_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train = scale.fit_transform(rX_train)
X_vali: np.ndarray = scale.transform(rX_vali)  # type:ignore
X_test: np.ndarray = scale.transform(rX_test)  # type:ignore

print(X_train.shape, X_vali.shape, X_test.shape)

# Delete these generic variables that we really shouldn't use anymore now that the data is fully-prepared.
del X, y, ys, rX_train, rX_vali, rX_test

(N, D) = X_train.shape

accuracy_graphs = {}
auc_graphs = {}

best_score = 0.0
best_model = None
for i in range(1000):
    m = LinearModel.random(D)
    train_score = m.score(X_vali, y_vali)
    if train_score > best_score or best_model is None:
        best_score = train_score
        best_model = m
        print("rand[{}] = {:.3}".format(i, train_score))

print(["{:1.3f}".format(x[0]) for x in best_model.weights.tolist()])

accuracy_graphs["Random"] = bootstrap_accuracy(best_model, X_vali, y_vali)
auc_graphs["Random"] = bootstrap_auc(best_model, X_vali, y_vali)

for i in range(20):
    sgd = SGDClassifier(random_state=i + RANDOM_SEED)
    sgd.fit(X_train, y_train)
    train_score = sgd.score(X_vali, y_vali)
    if train_score > best_score or best_model is None:
        best_score = train_score
        best_model = sgd
        print("sgd[{}] = {:.3}".format(i, train_score))

accuracy_graphs["SGD"] = bootstrap_accuracy(best_model, X_vali, y_vali)
auc_graphs["SGD"] = bootstrap_auc(best_model, X_vali, y_vali)


def mini_ca():
    ## MINI-CA
    print("### MINI-CA ###\n\n")
    # Try baby-coordinate ascent:
    ca = LinearModel.random(D)
    print("ca.start = {:.3}".format(ca.score(X_train, y_train)))
    dims = list(range(D + 1))
    random.shuffle(dims)
    for d in dims:
        better = ca.improve_feature_weight(X_train, y_train, feature_id=d)
        if better:
            print(
                "ca.weights[{}] is better = {:.3}".format(d, ca.score(X_train, y_train))
            )


def ca_restart(loud=False, measure=lambda m, X, y: m.score(X, y)):
    ## MINI-CA
    if loud:
        print("### MINI-CA.v2 ###\n\n")
    # Try baby-coordinate ascent:
    ca = LinearModel.random(D)
    if loud:
        print("ca2.start = {:.3}".format(ca.score(X_train, y_train)))
    dims = list(range(D + 1))

    # keep optimizing until we stop getting better!
    while True:
        any_better = False
        random.shuffle(dims)
        for d in dims:
            better = ca.improve_feature_weight(
                X_train, y_train, feature_id=d, measure=measure
            )
            if better:
                any_better = True
        if not any_better:
            break
        if loud:
            print("w+ = {:.3}".format(ca.score(X_train, y_train)))
    return ca


best_score = 0.0
best_model = None
for i in range(20):
    ca = ca_restart()
    train_score = ca.score(X_vali, y_vali)
    if train_score > best_score or best_model is None:
        best_score = train_score
        best_model = ca
        print("ca[{}] = {:.3}".format(i, train_score))

accuracy_graphs["CoordinateAscent"] = bootstrap_accuracy(best_model, X_vali, y_vali)
auc_graphs["CoordinateAscent"] = bootstrap_auc(best_model, X_vali, y_vali)

do_slow_AUC_experiment = False
if do_slow_AUC_experiment:
    best_score = 0.0
    best_model = None
    for i in range(20):
        ca = ca_restart(measure=lambda m, X, y: m.compute_auc(X, y))
        train_score = ca.score(X_vali, y_vali)
        if train_score > best_score or best_model is None:
            best_score = train_score
            best_model = ca
            print("ca-AUC[{}] = {:.3}".format(i, train_score))
    accuracy_graphs["CoordinateAscent-AUC"] = bootstrap_accuracy(
        best_model, X_vali, y_vali
    )
    auc_graphs["CoordinateAscent-AUC"] = bootstrap_auc(best_model, X_vali, y_vali)

simple_boxplot(auc_graphs, "Linear Model AUC", save="graphs/p11-AUC.png")
simple_boxplot(accuracy_graphs, "Linear Model Accuracy", save="graphs/p11-Accuracy.png")