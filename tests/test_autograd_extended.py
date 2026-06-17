import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.nn import Value


def test_exp_gradient():

    x = Value(2.0)

    y = x.exp()

    y.backward()

    assert np.allclose(
        x.grad,
        np.exp(2.0)
    )
