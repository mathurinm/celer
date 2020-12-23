import os
import numpy as np
import pandas as pd
from scipy import sparse

from os.path import join as pjoin

from celer.datasets import CELER_PATH

BASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

NAMES = {'gisette_train': 'gisette/GISETTE/gisette_train'}


def fetch_ml_uci(dataset):
    """Get a datasest from ML UCI database.

    Parameters
    ----------
    dataset: string
        Dataset name. Must be in NAMES.keys()

    Returns
    -------
    X: np.array, shape (n_samples, n_features)
        Design matrix.
    y: np.array, shape (n_samples)
        Target vector.
    """
    if not os.path.exists(pjoin(CELER_PATH, "ml_uci")):
        os.makedirs(pjoin(CELER_PATH, "ml_uci"))

    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s" % dataset)

    X_path = pjoin(CELER_PATH, "ml_uci", dataset + '_data.npz')
    y_path = pjoin(CELER_PATH, "ml_uci", dataset + '_target.npy')
    try:
        X = sparse.load_npz(X_path)
        y = np.load(y_path)
    except FileNotFoundError:
        df = pd.read_csv(BASE + NAMES[dataset] + '.data', sep=' ', header=None)
        # trailing wspace > extra column
        X = sparse.csc_matrix(df.values[:, :-1])
        y = np.array(pd.read_csv(BASE + NAMES[dataset] + '.labels',
                                 header=None)).ravel()
        sparse.save_npz(X_path, X)
        np.save(y_path, y)

    return X, y
