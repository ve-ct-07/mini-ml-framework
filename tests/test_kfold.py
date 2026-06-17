import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.model_selection import KFold


def test_fold_sizes():

    X = np.arange(103)

    kf = KFold(n_splits=5)

    total = 0

    for _, test_idx in kf.split(X):
        total += len(test_idx)

    assert total == len(X)
