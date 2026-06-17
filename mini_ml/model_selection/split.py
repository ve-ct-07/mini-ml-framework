import numpy as np


def train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=None,
):
    """
    Split arrays into train and test sets.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain "
            "the same number of samples."
        )

    n_samples = len(X)

    if n_samples < 2:
        raise ValueError(
            "Need at least two samples."
        )

    if isinstance(test_size, float):

        if not (0.0 < test_size < 1.0):
            raise ValueError(
                "Float test_size must "
                "be in (0, 1)."
            )

        n_test = int(
            np.floor(test_size * n_samples)
        )

    elif isinstance(test_size, int):

        if not (0 < test_size < n_samples):
            raise ValueError(
                "Integer test_size "
                "out of range."
            )

        n_test = test_size

    else:
        raise TypeError(
            "test_size must be float or int."
        )

    n_train = n_samples - n_test

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(
            random_state
        )
        indices = rng.permutation(indices)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )

def train_test_val_split(
    X,
    y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    shuffle=True,
    random_state=None,
):
    """
    Split arrays into train,
    validation, and test sets.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain "
            "the same number of samples."
        )

    sizes = (
        train_size,
        val_size,
        test_size,
    )

    if not all(
        isinstance(s, float)
        for s in sizes
    ):
        raise TypeError(
            "All split sizes must be floats."
        )

    if not all(
        0.0 <= s <= 1.0
        for s in sizes
    ):
        raise ValueError(
            "All split sizes must "
            "be in (0,1)."
        )

    if not np.isclose(
        train_size + val_size + test_size,
        1.0,
    ):
        raise ValueError(
            "Split sizes must sum to 1."
        )

    n_samples = len(X)

    n_train = int(
        np.floor(train_size * n_samples)
    )

    n_val = int(
        np.floor(val_size * n_samples)
    )

    n_test = (
        n_samples
        - n_train
        - n_val
    )

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(
            random_state
        )
        indices = rng.permutation(indices)

    train_idx = indices[:n_train]

    val_idx = indices[
        n_train:n_train + n_val
    ]

    test_idx = indices[
        n_train + n_val:
    ]

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )
