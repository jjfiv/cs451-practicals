#%%
import random
from shared import bootstrap_accuracy, simple_boxplot
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import typing as T
from dataclasses import dataclass

#%%

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]

(N, D) = X_train.shape
#%% Train up Forest models:

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
print("Forest.score = {:.3}".format(forest.score(X_vali, y_vali)))

lr = LogisticRegression()
lr.fit(X_train, y_train)
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
graphs = {
    "RF": bootstrap_accuracy(forest, X_vali, y_vali),
    "SGD": bootstrap_accuracy(sgd, X_vali, y_vali),
    "LR": bootstrap_accuracy(lr, X_vali, y_vali),
}

#%% SVM
from sklearn.svm import SVC as SVMClassifier

configs = []
configs.append({"kernel": "linear"})
configs.append({"kernel": "poly", "degree": 2})
configs.append({"kernel": "poly", "degree": 3})
configs.append({"kernel": "rbf"})
# configs.append({"kernel": "sigmoid"}) # just awful.


@dataclass
class ModelInfo:
    name: str
    accuracy: float
    model: T.Any
    X_vali: T.Optional[np.ndarray] = None


# TODO: C is the most important value for a SVM.
#       1/C is how important the model stays small.
# TODO: RBF Kernel is the best; explore it's 'gamma' parameter.

for cfg in configs:
    variants: T.List[ModelInfo] = []
    for class_weights in [None, "balanced"]:
        for c_val in [1.0]:
            svm = SVMClassifier(C=c_val, class_weight=class_weights, **cfg)
            svm.fit(X_train, y_train)
            name = "k={}{} C={} {}".format(
                cfg["kernel"], cfg.get("degree", ""), c_val, class_weights or ""
            )
            accuracy = svm.score(X_vali, y_vali)
            print("{}. score= {:.3}".format(name, accuracy))
            variants.append(ModelInfo(name, accuracy, svm))
    best = max(variants, key=lambda x: x.accuracy)
    graphs[best.name] = bootstrap_accuracy(best.model, X_vali, y_vali)


simple_boxplot(
    graphs,
    title="Kernelized Models for Poetry",
    ylabel="Accuracy",
    save="graphs/p15-kernel-cmp.png",
)