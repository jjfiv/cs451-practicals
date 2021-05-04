import random
import numpy as np
from dataclasses import dataclass, field
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import resample
import typing as T
from shared import TODO

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]


def y_to_sign(x: bool) -> int:
    if x:
        return 1
    else:
        return -1


@dataclass
class WeightedEnsemble(ClassifierMixin):
    """ A weighted ensemble is a list of (weight, classifier) tuples."""

    members: T.List[T.Tuple[float, T.Any]] = field(default_factory=list)

    def insert(self, weight: float, model: T.Any):
        self.members.append((weight, model))

    def predict_one(self, x: np.ndarray) -> bool:
        vote_sum = 0
        for weight, clf in self.members:
            vote_sum += y_to_sign(clf.predict([x])[0]) * weight
        return vote_sum > 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        (N, _) = X.shape
        class_votes = np.zeros((N, 1))
        for weight, clf in self.members:
            ys = clf.predict(X)
            for i, y in enumerate(ys):
                class_votes[i] += y_to_sign(y) * weight
        return class_votes > 0


#%%

tree_num = []
tree_vali = []
forest_vali = []

forest = WeightedEnsemble()
(N, D) = X_train.shape
for i in range(100):
    # bootstrap sample the training data
    X_sample, y_sample = resample(X_train, y_train)  # type:ignore

    # TODO create a tree model.
    tree = TODO("train and fit a model to the sampled data")

    # TODO Experiment:
    # What if instead of every tree having the same 1.0 weight, we considered some alternatives?
    #  - weight = the accuracy of that tree on the whole training set.
    #  - weight = the accuracy of that tree on the validation set.
    #  - weight = random.random()
    #  - weight = 0.1
    weight = 1.0

    # hold onto it for voting
    forest.insert(weight, tree)

    tree_num.append(i)
    tree_vali.append(tree.score(X_vali, y_vali))
    forest_vali.append(forest.score(X_vali, y_vali))
    if i % 5 == 0:
        print("Tree[{}] = {:.3}".format(i, tree_vali[-1]))
        print("Forest[{}] = {:.3}".format(i, forest_vali[-1]))

import matplotlib.pyplot as plt

plt.plot(tree_num, tree_vali, label="Individual Trees", alpha=0.5)
plt.plot(tree_num, forest_vali, label="Random Forest")
plt.legend()
plt.show()
