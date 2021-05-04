import random
import numpy as np
from dataclasses import dataclass, field
from numpy.core.fromnumeric import argmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import resample
import typing as T
import math

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]


@dataclass
class WeightedEnsemble(ClassifierMixin):
    """ A weighted ensemble is a list of (weight, classifier) tuples."""

    members: T.List[T.Tuple[float, T.Any]] = field(default_factory=list)

    def predict_one(self, x: np.ndarray) -> bool:
        vote_sum = 0
        for weight, clf in self.members:
            y = clf.predict([x])[0]
            if y:
                vote_sum += weight
            else:
                vote_sum -= weight
        return vote_sum > 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        (N, D) = X.shape
        class_votes = np.zeros((N, 1))
        for weight, clf in self.members:
            ys = clf.predict(X)
            for i, y in enumerate(ys):
                if y:
                    class_votes[i] += weight
                else:
                    class_votes[i] -= weight
        return class_votes > 0


def adaboost(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    learning_rate: float = 0.5,
    make_weak_learner=lambda: DecisionTreeClassifier(max_depth=4),
) -> WeightedEnsemble:
    (N, D) = X.shape
    importances = np.ones(N) / N  # uniform init
    output = WeightedEnsemble()
    # Train k classifiers
    for _ in range(k):
        # Train a weak learner:
        m = make_weak_learner()
        m.fit(X, y, sample_weight=importances)
        # Assess how it did:
        y_pred = m.predict(X)
        assert y_pred.shape == (N,)
        mistakes = np.asarray(y_pred != y, dtype=int)
        error = np.sum(mistakes * importances)
        # Stop if classification is perfect
        if error <= 0:
            break
        # compute importance of this learner
        alpha = learning_rate * (math.log((1.0 - error) - math.log(error)))
        output.members.append((alpha, m))
        # update importances:
        # importances *= np.exp(alpha * mistakes * (importances > 0))
        mistakes[mistakes == True] = -1
        mistakes[mistakes == False] = 1
        importances *= np.exp(-alpha * mistakes)
        sample_weight_sum = np.sum(importances)
        if sample_weight_sum <= 0:
            break
        importances /= sample_weight_sum
    return output


def make_dtree():
    return DecisionTreeClassifier(
        max_depth=4  # type:ignore
    )


def make_linear():
    return LogisticRegression()


#%% Train up AdaBoost models:

m = adaboost(X_train, y_train, 100, learning_rate=1.0, make_weak_learner=make_dtree)
print("Adaboost[DT].score = {:.3}".format(m.score(X_vali, y_vali)))
m = adaboost(X_train, y_train, 200, make_weak_learner=make_linear)
print("Adaboost[LR].score = {:.3}".format(m.score(X_vali, y_vali)))

# Don't make your own at home.
msk = AdaBoostClassifier(base_estimator=make_dtree(), n_estimators=100)
msk.fit(X_train, y_train)
print("Adaboost[sk].score = {:.3}".format(msk.score(X_vali, y_vali)))

# Generalization of Adaboost:
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
print("GBC[sk].score = {:.3}".format(gbc.score(X_vali, y_vali)))
