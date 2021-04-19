import numpy as np
from shared import (
    dataset_local_path,
)
from typing import Tuple, Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# start off by seeding random number generators:
RANDOM_SEED = 12345

df: pd.DataFrame = pd.read_json(dataset_local_path("poetry_id.jsonl"), lines=True)

features = pd.json_normalize(df.features)
features = features.join([df.poetry, df.words])

tv_f, test_f = train_test_split(features, test_size=0.25, random_state=RANDOM_SEED)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RANDOM_SEED)

textual = TfidfVectorizer(max_df=0.75, min_df=2, dtype=np.float32)
numeric = make_pipeline(DictVectorizer(sparse=False), StandardScaler())


def split(
    df: pd.DataFrame, fit: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    global numeric, textual
    y = np.array(df.pop("poetry").values)
    text = df.pop("words")
    if fit:
        textual.fit(text)
        numeric.fit(df.to_dict("records"))
    x_text = textual.transform(text)
    x_num = numeric.transform(df.to_dict("records"))
    x_merged = np.asarray(np.hstack([x_num, x_text.todense()]))
    return (y, {"textual": x_text, "numeric": x_num, "merged": x_merged})


y_train, Xd_train = split(train_f, fit=True)
y_vali, Xd_vali = split(vali_f)
y_test, Xd_test = split(test_f)
