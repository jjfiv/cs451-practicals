#%%
import numpy as np
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from typing import List
import random
from shared import bootstrap_auc
import matplotlib.pyplot as plt

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]

#%%
from sklearn.linear_model import LogisticRegression

m = LogisticRegression(random_state=RANDOM_SEED, penalty="none", max_iter=2000)
m.fit(X_train, y_train)

print("skLearn-LR AUC: {:.3}".format(np.mean(bootstrap_auc(m, X_vali, y_vali))))
print("skLearn-LR Acc: {:.3}".format(m.score(X_vali, y_vali)))


def nearly_eq(x, y, tolerance=1e-6):
    return abs(x - y) < tolerance


#%%
import torch
import torch.nn
import torch.optim


class SigmoidClassifier(torch.nn.Module):
    def __init__(self, D, num_classes=2):
        super(SigmoidClassifier, self).__init__()
        self.weights = torch.nn.Linear(D, num_classes, bias=True)

    def forward(self, X):
        return self.weights(X)


(N, D) = X_train.shape

X = torch.from_numpy(X_train).float()
y = torch.from_numpy(y_train).long()
Xv = torch.from_numpy(X_vali).float()
yv = torch.from_numpy(y_vali).long()


def train(name: str, model, optimizer, objective, max_iter=5000):
    train_losses = []
    vali_losses = []
    samples = []
    for it in tqdm(range(max_iter)):
        model.train()

        # Perform one step of training:
        optimizer.zero_grad()
        loss = objective(model(X), y)
        loss.backward()
        optimizer.step()

        if it % 25 == 0:
            model.eval()
            y_probs = model(X).detach().numpy()
            vali_loss = objective(model(Xv), yv)
            train_losses.append(loss.item())
            vali_losses.append(vali_loss.item())
            samples.append(it)
    model.eval()

    y_probs = model(Xv).detach().numpy()
    y_pred = (y_probs[:, 1] > 0.5).ravel()
    print(
        "Validation. Acc: {:.3} Auc: {:.3}".format(
            metrics.accuracy_score(yv, y_pred),
            metrics.roc_auc_score(yv, y_probs[:, 1].ravel()),
        )
    )

    plt.plot(samples, train_losses, label="Training Loss", alpha=0.7)
    plt.plot(samples, vali_losses, label="Validation Loss", alpha=0.7)
    plt.title("{} Training Loss".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/p13-{}-loss.png".format(name))
    plt.show()

    return model


model = SigmoidClassifier(D)
objective = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

train("logistic-regression", model, optimizer, objective)

DROPOUT = 0.2


def make_neural_net(D: int, hidden: List[int], num_classes: int = 2):
    layers = []
    for i, dim in enumerate(hidden):
        if i == 0:
            layers.append(torch.nn.Linear(D, dim))
            layers.append(torch.nn.Dropout(p=DROPOUT))
            layers.append(torch.nn.ReLU())
        else:
            layers.append(torch.nn.Linear(hidden[i - 1], dim))
            layers.append(torch.nn.Dropout(p=DROPOUT))
            layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden[-1], num_classes))
    return torch.nn.Sequential(*layers)


LEARNING_RATE = 0.1
MOMENTUM = 0.9
REGULARIZATION = 0.0  # try 0.1, 0.01, etc.

# two hidden layers, 16 nodes, each.
model = make_neural_net(D, [16, 16])
objective = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=REGULARIZATION
)

train("neural_net", model, optimizer, objective)
