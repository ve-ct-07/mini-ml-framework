"""
Dataset loading and synthetic data generation utilities.
"""

from .loaders import (
    DatasetNotFoundError,
    load_fashion_mnist,
    load_spambase,
)
from .synthetic import make_noisy_sine

__all__ = [
    "DatasetNotFoundError",
    "load_spambase",
    "load_fashion_mnist",
    "make_noisy_sine",
]
