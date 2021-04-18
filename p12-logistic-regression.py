#%%
from dataclasses import dataclass, field
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from shared import (
    dataset_local_path,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from typing import List, Tuple, Dict, Optional
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from scipy.sparse import hstack
from scipy.special import expit

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#%%

from shared import bootstrap_auc, dataset_local_path

df: pd.DataFrame = pd.read_json(dataset_local_path("poetry_id.jsonl"), lines=True)

features = pd.json_normalize(df.features)
features = features.join([df.poetry, df.words])
features.head()

tv_f, test_f = train_test_split(features, test_size=0.25, random_state=RANDOM_SEED)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RANDOM_SEED)

textual = TfidfVectorizer(max_df=0.75, min_df=2, dtype=np.float32)

numeric = make_pipeline(DictVectorizer(sparse=False), StandardScaler())


def split(df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    global numeric, textual
    y = np.array(df.pop("poetry").values)
    text = df.pop("words")
    if fit:
        textual.fit(text)
        numeric.fit(df.to_dict("records"))
    x_text = textual.transform(text)
    x_num = numeric.transform(df.to_dict("records"))
    x_merged = np.asarray(np.hstack([x_num, x_text.todense()]))
    return (y, x_num)


y_train, X_train = split(train_f, fit=True)
y_vali, X_vali = split(vali_f)

print(X_train.shape)
print(X_vali.shape)

#%%
from sklearn.linear_model import LogisticRegression

m = LogisticRegression(random_state=RANDOM_SEED, penalty="none", max_iter=2000)
m.fit(X_train, y_train)

print("skLearn-LR AUC: {:.3}".format(np.mean(bootstrap_auc(m, X_vali, y_vali))))
print("skLearn-LR Acc: {:.3}".format(m.score(X_vali, y_vali)))


@dataclass
class LogisticRegressionModel:
    # Managed to squeeze bias into this weights array by adding some +1s.
    weights: np.ndarray

    @staticmethod
    def random(D: int) -> "LogisticRegressionModel":
        weights = np.random.randn(D + 1, 1)
        return LogisticRegressionModel(weights)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """ Compute the expit of the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        assert self.weights[:D].shape == (D, 1)
        # Matrix multiplication; sprinkle transpose and assert to get the shapes you want (or remember Linear Algebra)... or both!
        output = np.dot(self.weights[:D].transpose(), X.transpose())
        assert output.shape == (1, N)
        # now add bias and put it through the 'expit' function.
        return np.array(expit(output + self.weights[-1])).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.decision_function(X) > 0.5, dtype="int32").ravel()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Take predictions and compute accuracy. """
        y_hat = self.predict(X)
        return metrics.accuracy_score(np.asarray(y), y_hat)  # type:ignore


@dataclass
class ModelTrainingCurve:
    train: List[float] = field(default_factory=list)
    validation: List[float] = field(default_factory=list)

    def add_sample(
        self,
        m: LogisticRegressionModel,
        X: np.ndarray,
        y: np.ndarray,
        X_vali: np.ndarray,
        y_vali: np.ndarray,
    ) -> None:
        self.train.append(m.score(X, y))
        self.validation.append(m.score(X_vali, y_vali))


(N, D) = X_train.shape

learning_curves: Dict[str, ModelTrainingCurve] = {}


def train_linear_regression_gd(name: str, num_iter=100):
    plot = ModelTrainingCurve()
    learning_curves[name] = plot

    m = LogisticRegressionModel.random(D)
    m.weights[D] = np.mean(y_train)
    alpha = 0.1

    pbar = tqdm(range(num_iter), total=num_iter, desc=name)
    for _iter in pbar:
        y_hat = m.decision_function(X_train)
        y_diffs = y_hat - y_train

        # look at all the predictions to compute our derivative:
        gradient = np.zeros((D + 1, 1))
        for i in range(N):
            x_vec = X_train[i, :].reshape((D, 1))
            y_err = float(y_diffs[i])
            # update weights; each instance is weighted by its err:
            gradient[:D] += y_err * x_vec
            gradient[D] += y_err * 1  # bias is always 1

        # take an alpha-sized step in the negative direction ('down')
        m.weights += -alpha * (gradient / N)

        # record performance:
        plot.add_sample(m, X_train, y_train, X_vali, y_vali)


def compute_gradient_update(m, X, y) -> np.ndarray:
    (N, D) = X.shape
    y_hat = m.decision_function(X)
    y_diffs = np.array(y_hat - y)
    # look at all the predictions to compute our derivative:
    gradient = np.zeros((D + 1, 1))

    # Literally a bajillion times faster if we ditch the for loops!
    # 1. scale X matrix by the y_diffs; then sum columns:
    x_scaled_by_y = X.T * y_diffs
    non_bias_gradient = np.sum(x_scaled_by_y, axis=1)
    gradient[:D] = non_bias_gradient.reshape((D, 1))
    # 2. the bias term is always 1 in X rows; so we just sum up the y_diffs for this.
    gradient[D] += np.sum(y_diffs)

    # take an gradient step in the negative direction ('down')
    return -(gradient / N)


def train_linear_regression_gd_opt(name: str, num_iter=100):
    plot = ModelTrainingCurve()
    learning_curves[name] = plot

    m = LogisticRegressionModel.random(D)
    # Alpha is the 'learning rate'.
    alpha = 0.1

    for iteration in tqdm(range(num_iter), total=num_iter, desc=name):
        # Each step is scaled by alpha, to control how fast we move, overall:
        m.weights += alpha * compute_gradient_update(m, X_train, y_train)
        # record performance:
        plot.add_sample(m, X_train, y_train, X_vali, y_vali)
    return m


def ideal_sgd(name: str, num_iter=100):
    plot = ModelTrainingCurve()
    learning_curves[name] = plot

    m = LogisticRegressionModel.random(D)
    # Alpha is the 'learning rate'.
    alpha = 0.1

    for iteration in tqdm(range(num_iter), total=num_iter, desc=name):
        for i in range(N):
            # Each step is scaled by alpha, to control how fast we move, overall:
            m.weights += alpha * compute_gradient_update(m, X_train[i, :], y_train[i])
        # record performance:
        plot.add_sample(m, X_train, y_train, X_vali, y_vali)
    return m


m = train_linear_regression_gd_opt("LR-GD", num_iter=2000)
print("LR-GD AUC: {:.3}".format(np.mean(bootstrap_auc(m, X_vali, y_vali))))
print("LR-GD Acc: {:.3}".format(m.score(X_vali, y_vali)))


def shuffle_slide_sgd(name, plot, num_iter, minibatch_size=512):
    m = LogisticRegressionModel.random(D)
    alpha = 0.1

    order = list(range(N))
    for _iter in tqdm(range(num_iter), total=num_iter, desc=name):
        # shuffle:
        random.shuffle(order)
        # loop over the size of the dataset in steps of 512
        for start in range(0, N, minibatch_size):
            # use the next contiguous bit:
            minibatch_end = min(N, start + minibatch_size)
            mb_idx = np.array([order[i] for i in range(start, minibatch_end)])
            # our sampled dataset:
            X_mb = X_train[mb_idx, :]
            y_mb = y_train[mb_idx]
            # update weights
            m.weights += alpha * compute_gradient_update(m, X_mb, y_mb)

        # record performance:
        plot.add_sample(m, X_train, y_train, X_vali, y_vali)


def train_linear_regression_sgd_opt(name: str, num_iter=100, minibatch_size=512):
    plot = ModelTrainingCurve()
    learning_curves[name] = plot

    m = LogisticRegressionModel.random(D)
    alpha = 0.1

    n_samples = max(1, N // minibatch_size)

    for _ in tqdm(range(num_iter), total=num_iter, desc=name):
        for _ in range(n_samples):
            X_mb, y_mb = resample(X_train, y_train, n_samples=minibatch_size)
            m.weights += alpha * compute_gradient_update(m, X_mb, y_mb)
        # record performance:
        plot.add_sample(m, X_train, y_train, X_vali, y_vali)
    return m


m = train_linear_regression_sgd_opt("LR-SGD", num_iter=2000)
print("LR-SGD AUC: {:.3}".format(np.mean(bootstrap_auc(m, X_vali, y_vali))))
print("LR-SGD Acc: {:.3}".format(m.score(X_vali, y_vali)))

import matplotlib.pyplot as plt


for key, dataset in learning_curves.items():
    xs = np.array(list(range(len(dataset.train))))
    plt.plot(xs, dataset.train, label="{}".format(key), alpha=0.7)
plt.title("Training Curves")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/p12-training-curves.png")
plt.show()

for key, dataset in learning_curves.items():
    xs = np.array(list(range(len(dataset.validation))))
    plt.plot(xs, dataset.validation, label="{}".format(key), alpha=0.7)
plt.title("Validation Curves")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/p12-vali-curves.png")
plt.show()
