# This 'shared.py' will be imported from in our practicals going forward (sometimes, anyway).
import os
import urllib.request
import sys


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
