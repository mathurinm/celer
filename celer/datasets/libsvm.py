# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import os
from bz2 import BZ2Decompressor
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from download import download
from scipy import sparse
import numpy as np

from os.path import join as pjoin
from pathlib import Path
CELER_PATH = pjoin(str(Path.home()), 'celer_data')


NAMES = {'rcv1_train': 'binary/rcv1_train.binary',
         'news20': 'binary/news20.binary',
         'finance': 'regression/log1p.E2006.train',
         'kdda_train': 'binary/kdda'}

N_FEATURES = {'finance': 4272227,
              'news20': 1355191,
              'rcv1_train': 47236,
              'kdda_train': 20216830}


def download_libsvm(dataset, destination, replace=False):
    """Download a dataset from LIBSVM website."""
    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s" % dataset)
    url = ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/" +
           NAMES[dataset])
    path = download(url + '.bz2', destination, replace=replace)
    return path


def decompress_data(compressed_path, decompressed_path):
    """Decompress a Libsvm dataset."""
    decompressor = BZ2Decompressor()
    with open(decompressed_path, "wb") as f, open(compressed_path, "rb") as g:
        for data in iter(lambda: g.read(100 * 1024), b''):
            f.write(decompressor.decompress(data))


def preprocess_libsvm(dataset, decompressed_path, X_path, y_path,
                      is_regression=False):
    """Preprocess a LIBSVM dataset."""
    # Normalization performed:
    # - X with only columns with >= 3 non zero elements, norm-1 columns
    # - y centered and set to std equal to 1
    # """
    n_features_total = N_FEATURES[dataset]
    with open(decompressed_path, 'rb') as f:
        X, y = load_svmlight_file(f, n_features_total)
        X = sparse.csc_matrix(X)

        NNZ = np.diff(X.indptr)  # number of non zero elements per feature
        # keep only features with >=3 non zero values
        X_new = X[:, NNZ >= 3]

        # set all feature norms to 1
        X_new = preprocessing.normalize(X_new, axis=0)
        if is_regression:
            # center y
            y -= np.mean(y)
            # normalize y to get a first duality gap of 0.5
            y /= np.std(y)

        # very important for sparse/sparse dot products: have sorted X.indices
        X_new.sort_indices()
        sparse.save_npz(X_path, X_new)
        np.save(y_path, y)


def download_preprocess_libsvm(dataset, replace=False, repreprocess=False):
    """Download and preprocess a given libsvm dataset."""

    paths = [CELER_PATH, pjoin(CELER_PATH, 'regression'),
             pjoin(CELER_PATH, 'binary'),
             pjoin(CELER_PATH, 'preprocessed')]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    print("Dataset: %s" % dataset)
    compressed_path = pjoin(CELER_PATH, "%s.bz2" % NAMES[dataset])
    download_libsvm(dataset, compressed_path, replace=replace)

    decompressed_path = pjoin(CELER_PATH, "%s" % NAMES[dataset])
    if not os.path.isfile(decompressed_path):
        decompress_data(compressed_path, decompressed_path)

    y_path = pjoin(CELER_PATH, "preprocessed", "%s_target.npy" % dataset)
    X_path = pjoin(CELER_PATH, "preprocessed", "%s_data.npz" % dataset)

    if (repreprocess or not os.path.isfile(y_path) or
            not os.path.isfile(X_path)):
        print("Preprocessing...")
        preprocess_libsvm(dataset, decompressed_path, X_path, y_path)


def load_libsvm(dataset):
    try:
        X = sparse.load_npz(pjoin(CELER_PATH, 'preprocessed',
                                  '%s_data.npz' % dataset))
        y = np.load(pjoin(CELER_PATH, 'preprocessed',
                          '%s_target.npy' % dataset))
    except FileNotFoundError:
        download_preprocess_libsvm(dataset, replace=False, repreprocess=True)
        X = sparse.load_npz(pjoin(CELER_PATH, 'preprocessed',
                                  '%s_data.npz' % dataset))
        y = np.load(pjoin(CELER_PATH, 'preprocessed',
                          '%s_target.npy' % dataset))
    return X, y


if __name__ == "__main__":
    for dataset in NAMES:
        download_preprocess_libsvm(dataset, replace=False, repreprocess=False)
