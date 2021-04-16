#%%
import pandas as pd
import numpy as np
import typing as T
import re
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from shared import bootstrap_auc, dataset_local_path, simple_boxplot

df: pd.DataFrame = pd.read_json(
    dataset_local_path("lit-wiki-2020.jsonl.gz"), lines=True
)

# Debug loading:
# df.head()


# Regular expresssions to grab parts of the text:
WORDS = re.compile(r"(\w+)")
NUMBERS = re.compile(r"(\d+)")


def extract_features(row):
    """
    Given the title and body of a Wikipedia article,
    extract features that might be of use to the 'is literary' task.

    Return named features in a dictionary.
    """
    title = row["title"].lower()
    body = row["body"]

    new_features: T.Dict[str, T.Any] = {}
    words = WORDS.findall(body)
    numbers = [int(x) for x in NUMBERS.findall(body)]

    new_features = {
        "page_rank": row["page_rank"],
        "list": "list" in title,
        "disambig": "disambiguation" in title,
        "maybe_person": "born" in words or "died" in words,
        "length": len(words),
        "num_categories": body.count("Category:"),
        "count_numbers": len(numbers),
        "count<1700": sum(1 for x in numbers if 1000 <= x <= 1700),
        "count<1800": sum(1 for x in numbers if 1700 < x <= 1800),
        "count<1900": sum(1 for x in numbers if 1800 < x <= 1900),
        "count<1920": sum(1 for x in numbers if 1900 < x <= 1920),
        "count<2100": sum(1 for x in numbers if 1920 < x <= 2100),
    }
    if len(numbers) > 0:
        new_features["mean_n"] = np.mean(numbers)
        new_features["std_n"] = np.std(numbers)

    return new_features


designed_f = pd.json_normalize(df.apply(extract_features, axis="columns"))

designed_f.head()
#%%


features: pd.DataFrame = designed_f.join([df.truth_value, df.body])
features = features.fillna(0.0)
#%%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack

RAND = 1234

# split the whole dataframe:
tv_f, test_f = train_test_split(features, test_size=0.25, random_state=RAND)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RAND)

textual = TfidfVectorizer(
    ngram_range=(1, 1), max_df=0.75, min_df=2, strip_accents="unicode"
)

# All numeric features should be normalized; use a pipeline here:
numeric = make_pipeline(DictVectorizer(sparse=False), StandardScaler())


def prepare_data(
    df: pd.DataFrame, fit: bool = False
) -> T.Tuple[np.ndarray, T.Dict[str, np.ndarray]]:
    global numeric, textual
    # extract truth_value as y:
    y = df.pop("truth_value").values
    # extract body as text:
    text = df.pop("body")
    if fit:
        # if this is the training data, fit it
        textual.fit(text)
        numeric.fit(df.to_dict("records"))
    # transform no matter what
    x_text = textual.transform(text)
    x_num = numeric.transform(df.to_dict("records"))
    # join pieces together:
    x_features = {
        "numeric": x_num,
        "textual": x_text,
        "merged": hstack([x_num, x_text]),
    }
    return y, x_features


# use the 'prepare_data'
train_y, train_xd = prepare_data(train_f, fit=True)
vali_y, vali_xd = prepare_data(vali_f)
test_y, test_xd = prepare_data(test_f)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Model:
    features: str
    m: T.Any
    train_score: float = 0.0
    vali_score: float = 0.0


models = []
for features in ["textual", "numeric", "merged"]:
    base_args = {"random_state": RAND, "class_weight": "balanced"}
    for d in range(4, 9):
        models.append(
            Model(
                features,
                RandomForestClassifier(max_depth=d, n_estimators=30, **base_args),
            )
        )
    models.append(Model(features, LogisticRegression(**base_args)))


for model in tqdm(models):
    model.m.fit(train_xd[model.features], train_y)
    model.train_score = model.m.score(train_xd[model.features], train_y)
    model.vali_score = model.m.score(vali_xd[model.features], vali_y)
    print(
        "{}/{}-Train-Acc: {:.3}, Vali-Acc: {:.3}".format(
            model.m.__class__.__name__,
            model.features,
            model.train_score,
            model.vali_score,
        )
    )

best_textual = max(
    [m for m in models if m.features == "textual"], key=lambda m: m.vali_score
)
best_numeric = max(
    [m for m in models if m.features == "numeric"], key=lambda m: m.vali_score
)
best_merged = max(
    [m for m in models if m.features == "merged"], key=lambda m: m.vali_score
)

train_sX = np.hstack(
    [
        best_textual.m.predict_proba(train_xd["textual"]),
        train_xd["numeric"],
    ]
)
vali_sX = np.hstack(
    [
        best_textual.m.predict_proba(vali_xd["textual"]),
        vali_xd["numeric"],
    ]
)
test_sX = np.hstack(
    [
        best_textual.m.predict_proba(test_xd["textual"]),
        test_xd["numeric"],
    ]
)


stacked = LogisticRegression(random_state=RAND)
stacked.fit((train_sX), train_y)

graphs = {
    "textual": bootstrap_auc(best_textual.m, test_xd["textual"], test_y),
    "numeric": bootstrap_auc(best_numeric.m, test_xd["numeric"], test_y),
    "merged": bootstrap_auc(best_merged.m, test_xd["merged"], test_y),
    "stacked": bootstrap_auc(stacked, test_sX, test_y),
}

simple_boxplot(
    graphs, ylabel="AUC", xlabel="method", save="graphs/p10-early-vs-stacked.png"
)

# %%
