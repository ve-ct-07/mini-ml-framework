"""
Dataset loading utilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested dataset file cannot be found."""


def load_spambase(
    data_folder: str = "data",
    filename: str = "spambase.data",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI Spambase dataset.

    Parameters
    ----------
    data_folder : str, default="data"
        Directory containing the dataset.
    filename : str, default="spambase.data"
        Dataset filename.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels.
    """
    file_path = Path(data_folder) / filename

    if not file_path.exists():
        raise DatasetNotFoundError(
            f"Spambase dataset not found: '{file_path}'."
        )

    data = np.loadtxt(file_path, delimiter=",")

    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.int64)

    return X, y


def load_fashion_mnist(
    data_folder: str = "data",
    train_filename: str = "fashion-mnist_train.csv",
    test_filename: str = "fashion-mnist_test.csv",
    kind: str = "train",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST from CSV files.

    Parameters
    ----------
    data_folder : str
        Dataset directory.
    train_filename : str
        Training CSV filename.
    test_filename : str
        Test CSV filename.
    kind : {"train", "test"}
        Dataset split to load.
    normalize : bool
        Whether to scale pixel values to [0, 1].

    Returns
    -------
    X : ndarray
        Flattened image vectors.
    y : ndarray
        Integer labels.
    """
    if kind not in {"train", "test"}:
        raise ValueError("kind must be either 'train' or 'test'")

    filename = train_filename if kind == "train" else test_filename
    file_path = Path(data_folder) / filename

    if not file_path.exists():
        raise DatasetNotFoundError(
            f"Fashion-MNIST file not found: '{file_path}'."
        )

    df = pd.read_csv(file_path)

    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)

    if normalize:
        X /= 255.0

    return X, y
