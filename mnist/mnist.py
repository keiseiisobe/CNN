import requests
from pathlib import Path
import gzip
import numpy as np
import os

base_dir = "mnist/data/"

mnist_files = {
    "x_train": "train-images-idx3-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz"
}

def _download():
    """
    download mnist datasets from remote only if they are not present in local.
    """
    base_url = "https://github.com/rossbar/numpy-tutorial-data-mirror/raw/refs/heads/main/"
    if not Path(base_dir).is_dir():
        os.mkdir(base_dir)
    for k, v in mnist_files.items():
        if Path(base_dir + v).exists():
            continue
        print("downloading", v, "...")
        r = requests.get(base_url + v, stream=True)
        with open(base_dir + v, "wb") as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)

def _decompress():
    mnist_datasets = {}
    for k, v in mnist_files.items():
        # x stands for images as input
        if "x" in k:
            with gzip.open(base_dir + v) as f:
                mnist_datasets[k] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        # for labels
        else:
            with gzip.open(base_dir + v) as f:
                mnist_datasets[k] = np.frombuffer(f.read(), np.uint8, offset=8)
    return mnist_datasets

def load_data():
    _download()
    mnist = _decompress()
    print("mnist datasets completely loaded")
    return mnist["x_train"], mnist["y_train"], mnist["x_test"], mnist["y_test"]
