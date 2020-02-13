import numpy as np

from celer.datasets import load_libsvm, load_climate
from celer.datasets.libsvm import NAMES


def test_random_download():
    dataset = np.random.choice(list(NAMES.keys()))
    print(dataset)
    X, y = load_libsvm(dataset)
    np.testing.assert_equal(X.shape[0], y.shape[0])


def test_climate():
    X, y = load_climate()
    np.testing.assert_equal(X.shape[0], y.shape[0])
