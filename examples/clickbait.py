from shared import dataset_local_path
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import os, gzip
from tqdm import tqdm

clickbait = pd.read_csv(dataset_local_path("clickbait.csv.gz"))

glove = {}
with gzip.open(os.environ["HOME"] + "/data/glove.6B.50d.txt.gz", "rt") as vecf:
    for line in tqdm(vecf, total=400000):
        split = line.index(" ")
        word = line[:split]
        vector = np.fromstring(line[split + 1 :], dtype=np.float32, sep=" ")
        glove[word] = vector
        if word == "the":
            print(word, vector)


print(clickbait.head())
# skip citation

df = clickbait.iloc[1:]

RANDOM_SEED = 12345

tv_f, test_f = train_test_split(df, test_size=0.25, random_state=RANDOM_SEED)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RANDOM_SEED)


def consider_tfidf():
    textual = TfidfVectorizer(max_df=0.75, min_df=1, dtype=np.float32)

    def prepare(df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        y = np.array(df.label.values)
        text = df.text.values
        if fit:
            textual.fit(text)
        x_text = textual.transform(text)
        return (y, x_text)

    y_train, X_train = prepare(train_f, fit=True)
    y_vali, X_vali = prepare(vali_f)
    y_test, X_test = prepare(test_f)

    m = SGDClassifier()
    m.fit(X_train, y_train)
    print("Train-Acc: {:.3}".format(m.score(X_train, y_train)))
    print("Vali-Acc: {:.3}".format(m.score(X_vali, y_vali)))

    word_order = sorted(zip(m.coef_.ravel().tolist(), textual.get_feature_names()))
    print("Ten least-clickbait words: ", word_order[:10])
    print("Ten most-clickbait words: ", word_order[-10:])


def consider_glove():
    textual = TfidfVectorizer()
    tokenizer = textual.build_analyzer()

    def prepare(df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        y = np.array(df.label.values)
        N = len(y)
        D = len(glove["the"])
        X = np.zeros((N, D))
        for i, example in enumerate(df.text):
            count = 0
            for word in tokenizer(example):
                if word in glove:
                    count += 1
                    X[i] += glove[word]
            if count > 0:
                X[i] /= count
        return (y, X)

    y_train, X_train = prepare(train_f, fit=True)
    y_vali, X_vali = prepare(vali_f)
    y_test, X_test = prepare(test_f)

    m = SGDClassifier()
    m.fit(X_train, y_train)
    print("glove-Train-Acc: {:.3}".format(m.score(X_train, y_train)))
    print("glove-Vali-Acc: {:.3}".format(m.score(X_vali, y_vali)))


consider_tfidf()
consider_glove()