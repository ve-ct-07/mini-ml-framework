"""
mini_ml: A foundational machine learning library built using NumPy.

Provides implementations of:
- Linear Models
- Naive Bayes
- Data Preprocessing
- Model Selection Utilities
- Neural Networks
"""

__version__ = "1.0.0"

# Import subpackages
from . import datasets
from . import linear_models
from . import model_selection
from . import naive_bayes
from . import preprocessing
from . import nn

__all__ = [
    "datasets",
    "linear_models",
    "model_selection",
    "naive_bayes",
    "preprocessing",
    "nn",
]
