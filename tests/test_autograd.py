import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.nn import Value


def test_square_gradient():

    x = Value(3.0)

    y = x ** 2

    y.backward()

    assert np.allclose(
        x.grad,
        6.0
    )
