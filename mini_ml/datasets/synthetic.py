"""
Synthetic dataset generators.
"""

import numpy as np


def make_noisy_sine(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a noisy sine-wave regression dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    noise : float
        Standard deviation of Gaussian noise.
    random_state : int or None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, 1)
    y : ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)

    X = np.linspace(
        0.0,
        2.0 * np.pi,
        n_samples,
        dtype=np.float64,
    ).reshape(-1, 1)

    y = np.sin(X[:, 0])
    y += rng.normal(0.0, noise, size=n_samples)

    return X, y
