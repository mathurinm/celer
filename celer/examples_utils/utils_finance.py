# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import os
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from download import download
from bz2 import BZ2Decompressor


def download_finance(path):
    """Download the Finance dataset from LIBSVM website."""
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets" \
        "/regression/log1p.E2006.train.bz2"
    path = download(url, path, replace=False)
    return path


def decompress_finance(compressed_path, decompressed_path):
    """Decompress the Finance dataset."""
    decompressor = BZ2Decompressor()
    with open(decompressed_path, "wb") as f, open(compressed_path, "rb") as g:
        for data in iter(lambda: g.read(100 * 1024), b''):
            f.write(decompressor.decompress(data))


def preprocess_finance(decompressed_path, X_path, y_path):
    """Preprocess the Finance dataset for a Lasso problem.

    Normalization performed:
    - X with only columns with >= 3 non zero elements, norm-1 columns, and
      a constant column to fit intercept
    - y centered and set to std equal to 1
    """
    n_features_total = 4272227
    with open(decompressed_path, 'rb') as f:
        X, y = load_svmlight_file(f, n_features_total)
        X = sparse.csc_matrix(X)

        NNZ = np.diff(X.indptr)  # number of non zero elements per feature
        # keep only features with >=3 non zero values
        X_new = X[:, NNZ >= 3]

        # set all feature norms to 1
        X_new = preprocessing.normalize(X_new, axis=0)
        # add constant feature to fit intercept
        X_new = preprocessing.add_dummy_feature(X_new, 1000.)
        # center y
        y -= np.mean(y)
        # normalize y to get a first duality gap of 0.5
        y /= np.std(y)

        # very important for sparse/sparse dot products: have sorted X.indices
        X_new.sort_indices()
        sparse.save_npz(X_path, X_new)
        np.save(y_path, y)


def download_preprocess_finance():
    """Download and preprocess the Finance dataset."""
    if not os.path.exists('./data'):
        os.mkdir('./data')

    compressed_path = "./data/log1p.E2006.train.bz2"
    download_finance(compressed_path)

    decompressed_path = "./data/log1p.E2006.train"
    if not os.path.isfile(decompressed_path):
        decompress_finance(compressed_path, decompressed_path)

    y_path = "./data/finance_target_preprocessed"
    X_path = "./data/finance_data_preprocessed"

    preprocess_finance(decompressed_path, X_path, y_path)


if __name__ == "__main__":
    download_preprocess_finance()
