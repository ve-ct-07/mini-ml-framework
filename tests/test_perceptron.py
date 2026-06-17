import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.linear_models.classification import (
    Perceptron
)


def test_linearly_separable():

    X = np.array([
        [1,1],
        [2,2],
        [3,3],
        [-1,-1],
        [-2,-2],
        [-3,-3]
    ])

    y = np.array([1,1,1,-1,-1,-1])

    model = Perceptron()

    model.fit(X,y)

    assert model.score(X,y) == 1.0
