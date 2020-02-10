# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import os
from os.path import join as pjoin
from pathlib import Path
from bz2 import BZ2Decompressor

import numpy as np
from scipy import sparse
from download import download
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file

CELER_PATH = pjoin(str(Path.home()), 'celer_data')


NAMES = {'rcv1_train': 'binary/rcv1_train.binary',
         'news20': 'binary/news20.binary',
         'finance': 'regression/log1p.E2006.train',
         'kdda_train': 'binary/kdda',
         'rcv1_topics_test': 'multilabel/rcv1_topics_test_2.svm'}

N_FEATURES = {'finance': 4272227,
              'news20': 1355191,
              'rcv1_train': 47236,
              'kdda_train': 20216830,
              'rcv1_topics_test': 47236}


def download_libsvm(dataset, destination, replace=False):
    """Download a dataset from LIBSVM website."""
    url = ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/" +
           NAMES[dataset])
    path = download(url + '.bz2', destination, replace=replace)
    return path


def decompress_data(compressed_path, decompressed_path):
    """Decompress a LIBSVM dataset."""
    decompressor = BZ2Decompressor()
    with open(decompressed_path, "wb") as f, open(compressed_path, "rb") as g:
        for data in iter(lambda: g.read(100 * 1024), b''):
            f.write(decompressor.decompress(data))


def preprocess_libsvm(dataset, decompressed_path, X_path, y_path,
                      is_regression=False):
    """Preprocess a LIBSVM dataset.
    Normalization performed:
    - X with only columns with >= 3 non zero elements, norm-1 columns
    - y centered and set to std equal to 1
    """
    n_features_total = N_FEATURES[dataset]
    with open(decompressed_path, 'rb') as f:
        multilabel = NAMES[dataset].split('/')[0] == 'multilabel'
        X, y = load_svmlight_file(f, n_features_total, multilabel=multilabel)
        X = sparse.csc_matrix(X)

        NNZ = np.diff(X.indptr)  # number of non zero elements per feature
        # keep only features with >=3 non zero values
        X_new = X[:, NNZ >= 3]

        # set all feature norms to 1
        # TODO this really makes no sense on a sparse matrix...
        X_new = preprocessing.normalize(X_new, axis=0)
        # very important for sparse/sparse dot products: have sorted X.indices
        X_new.sort_indices()
        sparse.save_npz(X_path, X_new)

        if multilabel:
            # Y is a row sparse matrix
            indices = np.array([lab for labels in y for lab in labels])
            indptr = np.cumsum([0] + [len(labels) for labels in y])
            data = np.ones_like(indices)
            # n_labels = np.max([np.max(labels) for labels in y])
            Y = sparse.csr_matrix((data, indices, indptr))
            sparse.save_npz(y_path, Y)

        else:
            if is_regression:
                # center y
                y -= np.mean(y)
                # normalize y to get a first duality gap of 0.5
                y /= np.std(y)
            np.save(y_path, y)


def download_preprocess_libsvm(dataset, replace=False, repreprocess=False):
    """Download and preprocess a given LIBSVM dataset."""

    paths = [CELER_PATH, pjoin(CELER_PATH, 'regression'),
             pjoin(CELER_PATH, 'binary'),
             pjoin(CELER_PATH, 'preprocessed')]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s" % dataset)

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
    """
    Download a dataset from LIBSVM website.

    Parameters
    ----------
    dataset : string
        Dataset name. Must be in celer.datasets.libsvm.NAMES.keys()

    Returns
    -------
    X : scipy.sparse.csc_matrix
        Design matrix.

    y : 1D or 2D np.array
        Design vector or matrix (in multiclass setting)


    References
    ----------
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

    """
    try:
        X = sparse.load_npz(pjoin(CELER_PATH, 'preprocessed',
                                  '%s_data.npz' % dataset))
        y = np.load(pjoin(CELER_PATH, 'preprocessed',
                          '%s_target.npy' % dataset))
    except FileNotFoundError:
        download_preprocess_libsvm(
            dataset, replace=False, repreprocess=True)
        X = sparse.load_npz(pjoin(CELER_PATH, 'preprocessed',
                                  '%s_data.npz' % dataset))
        y = np.load(pjoin(CELER_PATH, 'preprocessed',
                          '%s_target.npy' % dataset))
    return X, y


if __name__ == "__main__":
    for dataset in NAMES:
        download_preprocess_libsvm(
            dataset, replace=False, repreprocess=False)
