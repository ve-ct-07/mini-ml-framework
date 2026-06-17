import numpy as np


class KFold:
    """
    K-Fold cross-validator.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ):
        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):
        """
        Yield train and test indices.
        """
        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError(
                "n_splits cannot exceed number of samples."
            )

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(
                self.random_state
            )
            indices = rng.permutation(indices)

        fold_sizes = np.full(
            self.n_splits,
            n_samples // self.n_splits,
            dtype=int,
        )

        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0

        for fold_size in fold_sizes:

            start = current
            stop = current + fold_size

            test_idx = indices[start:stop]

            train_idx = np.concatenate(
                (
                    indices[:start],
                    indices[stop:],
                )
            )

            yield train_idx, test_idx

            current = stop

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ) -> int:
        return self.n_splits
