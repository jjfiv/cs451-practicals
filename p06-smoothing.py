from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import math

# new helpers:
from shared import (
    dataset_local_path,
    bootstrap_auc,
    simple_boxplot,
)

# stdlib:
from dataclasses import dataclass, field
import json, gzip
from typing import Dict, List


#%% load up the data
# Try 'POETRY'
dataset = "POETRY"
examples: List[str] = []
ys: List[bool] = []

if dataset == "WIKI":
    with gzip.open(dataset_local_path("lit-wiki-2020.jsonl.gz"), "rt") as fp:
        for line in fp:
            info = json.loads(line)
            # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
            keep = info["body"]
            # whether or not it's poetry is our label.
            ys.append(info["truth_value"])
            # hold onto this single dictionary.
            examples.append(keep)
else:
    # take only one per book!
    by_book = {}
    with open(dataset_local_path("poetry_id.jsonl")) as fp:
        for line in fp:
            info = json.loads(line)
            # dictionary keeps this key unique:
            by_book[info["book"]] = info
    # now extract only one page per book here:
    for info in by_book.values():
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["words"]
        # whether or not it's poetry is our label.
        ys.append(info["poetry"])
        # hold onto this single dictionary.
        examples.append(keep)

#%% Split data:

RANDOM_SEED = 1234

## split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    examples,
    ys,
    train_size=0.9,
    shuffle=True,
    random_state=RANDOM_SEED,
)
# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.9, shuffle=True, random_state=RANDOM_SEED
)

#%% Analyze Text

from sklearn.feature_extraction.text import CountVectorizer


# Note we're doing "CountVectorizer" here and not TfidfVectorizer. Hmm...
word_features = CountVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 1),
)

# How does it take a whole paragraph and turn it into words?
text_to_words = word_features.build_analyzer()
# text_to_words is a function (str) -> List[str]
assert text_to_words("Hello world!") == ["hello", "world"]

# Learn columns from training data (again)
word_features.fit(ex_train)
# Translate our list of texts -> matrices of counts
X_train = word_features.transform(ex_train)
X_vali = word_features.transform(ex_vali)
X_test = word_features.transform(ex_test)

print(X_train.shape, X_vali.shape, X_test.shape)

#%% Accumulate results here; to be box-plotted.
results: Dict[str, List[float]] = {}

#%% try sklearn MultinomialNB:

## SKLearn has it's own Multinomial Naive Bayes,
#  and it uses the alpha / additive smoothing to deal with zeros!
from sklearn.naive_bayes import MultinomialNB

# Try a couple alpha values (what to do with zero-prob words!)
# Alpha can really be anything positive!
for alpha in [0.1, 1.0, 10.0]:
    m = MultinomialNB(alpha=alpha)
    m.fit(X_train, y_train)
    scores = m.predict_proba(X_vali)[:, 1]
    print(
        "Accuracy: {:.3}, AUC: {:.3}".format(
            m.score(X_vali, y_vali), roc_auc_score(y_score=scores, y_true=y_vali)
        )
    )
    print("What I called log(beta)={}", m.class_log_prior_)
    results["MNB(alpha={})".format(alpha)] = bootstrap_auc(m, X_vali, y_vali)


#%% Showcase linar smoothing:

from collections import Counter
import typing


@dataclass
class CountLanguageModel:
    counts: typing.Counter[str] = field(default_factory=Counter)
    total: int = 0

    def add_example(self, words: List[str]) -> None:
        for word in words:
            self.counts[word] += 1
        self.total += len(words)

    def prob(self, word: str) -> float:
        return self.counts[word] / self.total


is_literary = CountLanguageModel()
is_random = CountLanguageModel()

# Train these two model pieces:
for y, ex in zip(y_train, ex_train):
    words = text_to_words(ex)
    # with linear smoothing, everything goes in random (positive OR negative)
    is_random.add_example(words)
    # but only positive go in positive:
    if y:
        is_literary.add_example(words)

print("lit: {}".format(is_literary.total))
print("rand: {}".format(is_random.total))

#
# The linear parameter is traditionally a non-zero, non-one probability:
#     (0 < linear < 1)
for linear in [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]:
    scores = []
    for ex in ex_vali:
        score = 0.0
        denom = 0
        for word in text_to_words(ex):
            prob_positive = is_literary.prob(word)
            prob_negative = is_random.prob(word)
            if prob_positive == 0.0 and prob_negative == 0.0:
                continue
            smoothed_positive = (prob_positive * linear) + (
                prob_negative * (1 - linear)
            )
            score += math.log(smoothed_positive) - math.log(prob_negative)
            denom += 1
        scores.append(score)
    print(roc_auc_score(y_score=scores, y_true=y_vali))
    # bootstrap AUC:
    dist = []
    # do the bootstrap:
    for trial in range(100):
        sample_pred, sample_truth = resample(
            scores, y_vali, random_state=trial + RANDOM_SEED
        )  # type:ignore
        score = roc_auc_score(y_true=sample_truth, y_score=sample_pred)  # type:ignore
        dist.append(score)
    results["Linear[{}]".format(linear)] = dist


#%% Boxplot all AUC results:
simple_boxplot(results, ylabel="AUC", save="{}-text-AUC.png".format(dataset))
