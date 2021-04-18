#%%
from dataclasses import dataclass, field
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from shared import (
    dataset_local_path,
)
from tqdm import tqdm
import random
from typing import List
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load the AirQualityUCI Dataset:
df = pd.read_csv(dataset_local_path("AirQualityUCI.csv"), sep=";", decimal=",")
print(df.shape)
# drop empty columns:
df = df.dropna(how="all", axis="columns")
print(df.shape)

PREDICT_COL = "CO(GT)"

# select only the rows where our 'y' is present:
df = df[df[PREDICT_COL] > -200.0]
print(df.shape)

# delete Date/Time columns
df.pop("Date")
df.pop("Time")

print(df.head())

#%%
#  Now train/test split:
tv_f, test_f = train_test_split(df, test_size=0.25, random_state=RANDOM_SEED)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RANDOM_SEED)

y_train = np.array(train_f.pop(PREDICT_COL).array)
y_vali = np.array(vali_f.pop(PREDICT_COL).array)
y_test = np.array(test_f.pop(PREDICT_COL).array)

#%%
# Now process data:
# Note, we don't NEED DictVectorizer... why?

# Let's fix missing values;
fix_missing = SimpleImputer(missing_values=-200.0)

scaler = StandardScaler()

X_train = scaler.fit_transform(fix_missing.fit_transform(train_f))
X_vali = scaler.transform(fix_missing.transform(vali_f))
X_test = scaler.transform(fix_missing.transform(test_f))


@dataclass
class LinearRegressionModel:
    # Managed to squeeze bias into this weights array by adding some +1s.
    weights: np.ndarray

    @staticmethod
    def random(D: int) -> "LinearRegressionModel":
        weights = np.random.randn(D + 1, 1)
        return LinearRegressionModel(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        assert self.weights[:D].shape == (D, 1)
        # Matrix multiplication; sprinkle transpose and assert to get the shapes you want (or remember Linear Algebra)... or both!
        output = np.dot(self.weights[:D].transpose(), X.transpose())
        assert output.shape == (1, N)
        return (output + self.weights[-1]).reshape((N,))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Take predictions and compute accuracy. """
        y_hat = self.predict(X)
        return float(metrics.r2_score(np.asarray(y), y_hat))  # type:ignore


@dataclass
class ModelTrainingCurve:
    train: List[float] = field(default_factory=list)
    validation: List[float] = field(default_factory=list)

    def add_sample(
        self,
        m: LinearRegressionModel,
        X: np.ndarray,
        y: np.ndarray,
        X_vali: np.ndarray,
        y_vali: np.ndarray,
    ) -> None:
        self.train.append(m.score(X, y))
        self.validation.append(m.score(X_vali, y_vali))


(N, D) = X_train.shape

learning_curves = {}


def train_linear_regression_gd(name: str, num_iter=100):
    plot = ModelTrainingCurve()
    learning_curves[name] = plot

    m = LinearRegressionModel.random(D)
    m.weights[D] = np.mean(y_train)
    alpha = 0.1

    pbar = tqdm(range(num_iter), total=num_iter, desc=name)
    for _iter in pbar:
        y_hat = m.predict(X_train)
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
    y_hat = m.predict(X)
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

    m = LinearRegressionModel.random(D)
    m.weights[D] = np.mean(y_train)
    alpha = 0.1

    pbar = tqdm(range(num_iter), total=num_iter, desc=name)
    for _iter in pbar:
        # take an alpha-sized step in the negative direction ('down')
        m.weights += alpha * compute_gradient_update(m, X_train, y_train)
        # record performance:
        plot.add_sample(m, X_train, y_train, X_vali, y_vali)
    return m


m = train_linear_regression_gd_opt("LR-GD", num_iter=500)
print("LR-GD R**2 {:.3}".format(m.score(X_vali, y_vali)))


def train_linear_regression_sgd_opt(name: str, num_iter=100, minibatch_size=512):
    plot = ModelTrainingCurve()
    learning_curves[name] = plot

    m = LinearRegressionModel.random(D)
    m.weights[D] = np.mean(y_train)
    alpha = 0.1

    n_samples = max(1, N // minibatch_size)

    for _ in tqdm(range(num_iter), total=num_iter, desc=name):
        # loop over the size of the dataset in steps of 512
        for _ in range(n_samples):
            # our sampled dataset:
            X_mb, y_mb = resample(X_train, y_train, n_samples=minibatch_size)
            m.weights += alpha * compute_gradient_update(m, X_mb, y_mb)

        # record performance:
        try:
            plot.add_sample(m, X_train, y_train, X_vali, y_vali)
        except ValueError:
            print(m.weights)


train_linear_regression_sgd_opt("LR-SGD", num_iter=500)
print("LR-GD R**2 {:.3}".format(m.score(X_vali, y_vali)))

import matplotlib.pyplot as plt

for key, dataset in learning_curves.items():
    xs = np.array(list(range(len(dataset.train))))
    plt.plot(xs, dataset.train, label="{}".format(key), alpha=0.7)
plt.title("Training Curves")
plt.xlabel("Iteration")
plt.ylabel("R**2")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/p12-training-r2.png")
plt.show()

for key, dataset in learning_curves.items():
    xs = np.array(list(range(len(dataset.validation))))
    plt.plot(xs, dataset.validation, label="{}".format(key), alpha=0.7)
plt.title("Validation Curves")
plt.xlabel("Iteration")
plt.ylabel("R**2")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/p12-vali-r2.png")
plt.show()
