import numpy as np
from dataclasses import dataclass, field
from sklearn.base import ClassifierMixin
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
from typing import List, DefaultDict, Any
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel


@dataclass
class LinearModel(ClassifierMixin):
    weights: np.ndarray  # note we can't specify this is 1-dimensional
    bias: float = 0.0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        assert self.weights.shape == (D, 1)
        # Matrix multiplication; sprinkle transpose and assert to get the shapes you want (or remember Linear Algebra)... or both!
        output = np.dot(self.weights.transpose(), X.transpose())
        assert output.shape == (1, N)
        return (output + self.bias).reshape((N,))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Take whether the points are above or below our hyperplane as a prediction. """
        return self.decision_function(X) > 0


def train_perceptron(y, X, num_iter=100, seed=1231) -> LinearModel:
    rand = np.random.default_rng(seed)
    (num_examples, num_features) = X.shape
    assert len(y) == num_examples
    w = np.zeros((num_features, 1))
    b = 0.0
    indices = list(range(num_examples))
    for _ in tqdm(range(num_iter)):
        rand.shuffle(indices)
        wrong = 0
        for i in indices:
            if y[i]:
                y_val = 1
            else:
                y_val = -1

            x_i = X[i, :].reshape((num_features, 1))

            activation = np.dot(w.transpose(), x_i) + b
            if y[i] != (activation > 0):
                wrong += 1
                # we got it wrong! update!
                w += y_val * x_i
                b += y_val
        if wrong == 0:
            break
    return LinearModel(w, b)


order1 = False


class ExampleBasedLinearModel(ClassifierMixin):
    def __init__(self, train_X: np.ndarray, train_y: np.ndarray):
        self.examples = train_X
        (N, D) = train_X.shape
        self.ex_weights = np.zeros(N)  # init uniform
        self.bias = float(np.mean(train_y))
        self.avg_weights = np.zeros(N)
        self.avg_bias = 0.0

    def decision_function(self, X: np.ndarray, train=False) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        (Nx, Dx) = self.examples.shape
        assert D == Dx
        assert self.ex_weights.shape == (Nx,)

        if train:
            model = self.ex_weights
        else:
            model = self.avg_weights

        if order1:
            # weights are now a distribution over the training examples.
            weights = np.sum(
                model * self.examples.T, axis=1
            )  # weight our learning examples
            assert weights.shape == (D,)
            output = np.dot(weights, X.T)
            assert output.shape == (N,)
        else:
            # This is effectively a 'pariwise similiarty'.
            pairwise = np.matmul(self.examples, X.T)
            assert pairwise.shape == (Nx, N)
            # weight the rows by ex_weights; sum them up.
            weighted = model * pairwise.T
            output = np.sum(weighted, axis=1)
            assert output.shape == (N,)
        return (output + self.bias).reshape((N,))

    def predict(self, X: np.ndarray, train=False) -> np.ndarray:
        """ Take whether the points are above or below our hyperplane as a prediction. """
        return self.decision_function(X, train) > 0

    def perceptron_step(self, X: np.ndarray, y: np.ndarray) -> int:
        (num_examples, num_features) = X.shape
        assert len(y) == num_examples
        indices = list(range(num_examples))
        random.shuffle(indices)
        wrong = 0
        life = 1
        for i in indices:
            if y[i]:
                y_val = 1
            else:
                y_val = -1

            activation = self.predict(X[i, :].reshape(1, num_features), train=True)[0]
            if y[i] != activation:
                wrong += 1
                self.avg_weights += self.ex_weights * life
                self.avg_bias += self.bias * life
                life = 1
                self.ex_weights[i] += y_val
                self.bias += y_val
            else:
                life += 1

        return wrong


class RBFPerceptron(ClassifierMixin):
    def __init__(self, train_X: np.ndarray, train_y: np.ndarray):
        self.examples = train_X
        (N, D) = train_X.shape
        self.gamma = 1 / (D * train_X.var())
        self.kernel = rbf_kernel(self.examples, self.examples, gamma=self.gamma)
        self.ex_weights = np.ones(N) / N  # init uniform
        self.bias = float(np.mean(train_y))

    def decision_function(self, X: np.ndarray, train_i=-1) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        (Nx, Dx) = self.examples.shape
        assert D == Dx
        assert self.ex_weights.shape == (Nx,)

        if train_i == -1:
            # This is effectively a 'pariwise similiarty'.
            pairwise = rbf_kernel(self.examples, X, gamma=self.gamma)
            assert pairwise.shape == (Nx, N)
            # weight the rows by ex_weights; sum them up.
            weighted = self.ex_weights * pairwise.T
            output = np.sum(weighted, axis=1)
        else:
            # fast-path, for training:
            assert N == 1
            pairwise = self.kernel[:, train_i].reshape((Nx, 1))
            assert pairwise.shape == (Nx, N)
            # weight the rows by ex_weights; sum them up.
            weighted = self.ex_weights * pairwise.T
            output = np.sum(weighted, axis=1)
        assert output.shape == (N,)

        return (output + self.bias).reshape((N,))

    def predict(self, X: np.ndarray, train_i=-1) -> np.ndarray:
        """ Take whether the points are above or below our hyperplane as a prediction. """
        return self.decision_function(X, train_i) > 0

    def perceptron_step(self, X: np.ndarray, y: np.ndarray) -> int:
        (num_examples, num_features) = X.shape
        assert len(y) == num_examples
        indices = list(range(num_examples))
        random.shuffle(indices)
        wrong = 0
        for i in indices:
            if y[i]:
                y_val = 1
            else:
                y_val = -1

            activation = self.predict(X[i, :].reshape(1, num_features), train_i=i)[0]
            if y[i] != activation:
                wrong += 1
                self.ex_weights[i] += y_val
                self.bias += y_val
        return wrong


class RBFAvgPerceptron(ClassifierMixin):
    def __init__(self, train_X: np.ndarray, train_y: np.ndarray):
        self.examples = train_X
        (N, D) = train_X.shape
        self.gamma = 1 / (D * train_X.var())
        self.kernel = rbf_kernel(self.examples, self.examples, gamma=self.gamma)
        self.ex_weights = np.ones(N) / N  # init uniform
        self.avg_weights = np.zeros(N)
        self.avg_bias = 0.0
        self.bias = float(np.mean(train_y))

    def decision_function(self, X: np.ndarray, train_i=-1) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        (Nx, Dx) = self.examples.shape
        assert D == Dx
        assert self.ex_weights.shape == (Nx,)

        if train_i == -1:
            # This is effectively a 'pariwise similiarty'.
            pairwise = rbf_kernel(self.examples, X, gamma=self.gamma)
            assert pairwise.shape == (Nx, N)
            # weight the rows by ex_weights; sum them up.
            weighted = self.avg_weights * pairwise.T
            output = np.sum(weighted, axis=1) + self.avg_bias
        else:
            # fast-path, for training:
            assert N == 1
            pairwise = self.kernel[:, train_i].reshape((Nx, 1))
            assert pairwise.shape == (Nx, N)
            # weight the rows by ex_weights; sum them up.
            weighted = self.ex_weights * pairwise.T
            output = np.sum(weighted, axis=1) + self.bias
        assert output.shape == (N,)

        return output.reshape((N,))

    def predict(self, X: np.ndarray, train_i=-1) -> np.ndarray:
        """ Take whether the points are above or below our hyperplane as a prediction. """
        return self.decision_function(X, train_i) > 0

    def perceptron_step(self, X: np.ndarray, y: np.ndarray) -> int:
        (num_examples, num_features) = X.shape
        assert len(y) == num_examples
        indices = list(range(num_examples))
        random.shuffle(indices)
        wrong = 0
        life = 0
        for i in indices:
            if y[i]:
                y_val = 1
            else:
                y_val = -1

            activation = self.predict(X[i, :].reshape(1, num_features), train_i=i)[0]
            if y[i] != activation:
                wrong += 1
                self.avg_weights += self.ex_weights * life
                self.avg_bias += self.bias * life
                life = 1
                self.ex_weights[i] += y_val
                self.bias += y_val
            else:
                life += 1
        return wrong


# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]

(N, D) = X_train.shape


@dataclass
class ModelTrainingCurve:
    train: List[float] = field(default_factory=list)
    validation: List[float] = field(default_factory=list)

    def add_sample(
        self,
        m: Any,
        X: np.ndarray,
        y: np.ndarray,
        X_vali: np.ndarray,
        y_vali: np.ndarray,
    ) -> None:
        self.train.append(m.score(X, y))
        self.validation.append(m.score(X_vali, y_vali))


# These are the named lines that will be plotted:
learning_curves: DefaultDict[str, ModelTrainingCurve] = defaultdict(ModelTrainingCurve)


def train_averaged_perceptron(
    y, X, y_vali, X_vali, num_iter=100, seed=1231
) -> LinearModel:
    rand = np.random.default_rng(seed)
    (num_examples, num_features) = X.shape
    assert len(y) == num_examples
    w_avg = np.zeros((num_features, 1))
    b_avg = 0.0
    w = np.zeros((num_features, 1))
    b = 0.0
    current_correct = 0
    indices = list(range(num_examples))
    for iteration in tqdm(range(num_iter)):
        rand.shuffle(indices)
        wrong = 0
        for i in indices:
            if y[i]:
                y_val = 1
            else:
                y_val = -1

            x_i = X[i, :].reshape((num_features, 1))

            activation = np.dot(w.transpose(), x_i) + b
            if y[i] != (activation > 0):
                # update 'average' vector:
                w_avg += current_correct * w
                b_avg += current_correct * b
                current_correct = 0
                # update 'current' vector
                wrong += 1
                # we got it wrong! update!
                w += y_val * x_i
                b += y_val
            else:
                current_correct += 1
        if wrong == 0:
            break
        tmp = LinearModel(w_avg, b_avg)
        learning_curves["Averaged-Perceptron"].add_sample(tmp, X, y, X_vali, y_vali)
    return LinearModel(w_avg, b_avg)


model = train_averaged_perceptron(y_train, X_train, y_vali, X_vali, num_iter=200)
print("AP. Train-Accuracy: {:.3}".format(model.score(X_train, y_train)))
print("AP. Vali-Accuracy: {:.3}".format(model.score(X_vali, y_vali)))

rbfp = RBFAvgPerceptron(X_train, y_train)
for i in range(200):
    rbfp.perceptron_step(X_train, y_train)
    print("RBFP[{}].score={:.3}".format(i, rbfp.score(X_vali, y_vali)))
    learning_curves["RBF-Perceptron"].add_sample(rbfp, X_train, y_train, X_vali, y_vali)


for key, dataset in learning_curves.items():
    xs = np.array(list(range(len(dataset.train))))
    # line-plot:
    plt.plot(xs, dataset.train, label="{} Train".format(key), alpha=0.7)
    plt.plot(xs, dataset.validation, label="{} Validate".format(key), alpha=0.7)
    # scatter-plot: (maybe these look nicer to you?)
    # plt.scatter(xs, points, label=key, alpha=0.7, marker=".")
    plt.ylim((0.75, 1.0))
    plt.title("{} Learning Curves".format(key))
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/p15-{}-acc-curve.png".format(key))
    plt.show()
