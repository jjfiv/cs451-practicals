#%%
from dataclasses import dataclass, field
import numpy as np
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import random
from typing import List, Dict
from sklearn.utils import resample
from scipy.special import expit
from shared import bootstrap_auc

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
alpha = 0.1

model = SigmoidClassifier(D)

objective = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

X = torch.from_numpy(X_train).float()
y = torch.from_numpy(y_train).long()
for _ in range(500):
    model.train()
    optimizer.zero_grad()
    loss = objective(model(X), y)
    loss.backward()
    optimizer.step()

    model.eval()
    y_probs = model.forward(torch.from_numpy(X_vali).float()).detach().numpy()
    y_pred = (y_probs[:, 1] > 0.5).ravel()
    print(
        "Loss: {:.3}, Accuracy: {:.3}".format(
            loss.item(), metrics.accuracy_score(y_vali, y_pred)
        )
    )
