# This 'shared.py' will be imported from in our practicals going forward (sometimes, anyway).
import os
import urllib.request
import sys
import typing
from typing import List, Dict, Optional, Any

from sklearn.base import ClassifierMixin
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import random


def bootstrap_accuracy(
    f: ClassifierMixin,
    X,  # numpy array
    y,  # numpy array
    num_samples: int = 100,
    random_state: int = random.randint(0, 2 ** 32 - 1),
) -> List[float]:
    """
    Take the classifier ``f``, and compute it's bootstrapped accuracy over the dataset ``X``,``y``.
    Generate ``num_samples`` samples; and seed the resampler with ``random_state``.
    """
    dist: List[float] = []
    y_pred = f.predict(X)  # type:ignore (predict not on ClassifierMixin)
    # do the bootstrap:
    for trial in range(num_samples):
        sample_pred, sample_truth = resample(
            y_pred, y, random_state=trial + random_state
        )  # type:ignore
        score = accuracy_score(y_true=sample_truth, y_pred=sample_pred)  # type:ignore
        dist.append(score)
    return dist


def TODO(for_what: str) -> None:
    """Because crashing should be legible."""
    print("=" * 80)
    print("TODO:", for_what, file=sys.stderr)
    print("=" * 80)
    sys.exit(-1)


def __create_data_directory():
    os.makedirs("data", exist_ok=True)
    assert os.path.exists("data") and os.path.isdir("data")


def __download_file(url: str, path: str):
    # empty data files were mis-downloaded...
    if os.path.exists(path) and os.path.getsize(path) > 0:
        # don't download multiple times.
        return
    # try connecting before creating output file...
    with urllib.request.urlopen(url) as f:
        # create output file and download the rest.
        with open(path, "w") as out:
            out.write(f.read().decode("utf-8"))


def dataset_local_path(name: str) -> str:
    __create_data_directory()
    destination = os.path.join("data", name)
    if name == "forest-fires.csv":
        __download_file(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv",
            destination,
        )
    elif name == "poetry_id.jsonl":
        __download_file(
            "http://ciir.cs.umass.edu/downloads/poetry/id_datasets.jsonl", destination
        )
    else:
        raise ValueError("No such dataset... {}; should you git pull?".format(name))
    assert os.path.exists(destination)
    return destination


def test_download():
    import json

    lpath = dataset_local_path("poetry_id.jsonl")
    with open(lpath) as fp:
        first = json.loads(next(fp))
        assert first["book"] == "aceptadaoficialmente00gubirich"


def simple_boxplot(
    data: Dict[str, List[float]],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show: bool = True,
    save: Optional[str] = None,
) -> Any:
    """ Create a simple set of named boxplots. """
    import matplotlib.pyplot as plt

    box_names = []
    box_dists = []
    for (k, v) in data.items():
        box_names.append(k)
        box_dists.append(v)
    plt.boxplot(box_dists)
    plt.xticks(ticks=range(1, len(box_names) + 1), labels=box_names)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    return plt