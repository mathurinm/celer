import numpy as np
import pandas as pd
from scipy import sparse

BASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

NAMES = {'gisette_train': 'gisette/GISETTE/gisette_train'}


def load_ml_uci(dataset):
    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s" % dataset)

    df = pd.read_csv(BASE + NAMES[dataset] + '.data', sep=' ', header=None)
    X = sparse.csc_matrix(df.values[:, :-1])  # trailing wspace > extra column
    y = np.array(pd.read_csv(BASE + NAMES[dataset] + '.labels',
                             header=None)).ravel()

    return X, y
