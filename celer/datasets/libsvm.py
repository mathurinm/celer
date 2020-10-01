# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import os
from os.path import join as pjoin
from bz2 import BZ2Decompressor

import numpy as np
from scipy import sparse
from download import download
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file

from celer.datasets import CELER_PATH


NAMES = {'rcv1_train': 'binary/rcv1_train.binary',
         'news20': 'binary/news20.binary',
         'finance': 'regression/log1p.E2006.train',
         'kdda_train': 'binary/kdda',
         'rcv1_topics_test': 'multilabel/rcv1_topics_test_2.svm',
         'real-sim': 'binary/real-sim',
         'sector_train': 'multiclass/sector/sector',
         'sector_test': 'multiclass/sector/sector.t',
         'url': 'binary/url_combined',
         'webspam': 'binary/webspam_wc_normalized_trigram.svm'}

N_FEATURES = {'finance': 4272227,
              'news20': 1355191,
              'rcv1_train': 47236,
              'kdda_train': 20216830,
              'rcv1_topics_test': 47236,
              'real-sim': 20958,
              'sector_train': 55197,
              'sector_test': 55197,
              'url': 3231961,
              'webspam': 16609143}


def download_libsvm(dataset, destination, replace=False):
    """Download a dataset from LIBSVM website."""
    url = ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/" +
           NAMES[dataset])
    path = download(url + '.bz2', destination, replace=replace)
    return path


def get_X_y(dataset, compressed_path, multilabel, replace=False):
    """Load a LIBSVM dataset as sparse X and observation y/Y.
    If X and y already exists as npz and npy, they are not redownloaded unless
    replace=True."""

    ext = '.npz' if multilabel else '.npy'
    y_path = pjoin(CELER_PATH, "%s_target%s" % (NAMES[dataset], ext))
    X_path = pjoin(CELER_PATH, "%s_data.npz" % NAMES[dataset])
    if replace or not os.path.isfile(y_path) or not os.path.isfile(X_path):
        tmp_path = pjoin(CELER_PATH, "%s" % NAMES[dataset])

        decompressor = BZ2Decompressor()
        print("Decompressing...")
        with open(tmp_path, "wb") as f, open(compressed_path, "rb") as g:
            for data in iter(lambda: g.read(100 * 1024), b''):
                f.write(decompressor.decompress(data))

        n_features_total = N_FEATURES[dataset]
        print("Loading svmlight file...")
        with open(tmp_path, 'rb') as f:
            X, y = load_svmlight_file(
                f, n_features=n_features_total, multilabel=multilabel)

        os.remove(tmp_path)
        X = sparse.csc_matrix(X)
        X.sort_indices()
        sparse.save_npz(X_path, X)

        if multilabel:
            indices = np.array([lab for labels in y for lab in labels])
            indptr = np.cumsum([0] + [len(labels) for labels in y])
            data = np.ones_like(indices)
            Y = sparse.csr_matrix((data, indices, indptr))
            sparse.save_npz(y_path, Y)
            return X, Y

        else:
            np.save(y_path, y)

    else:
        X = sparse.load_npz(X_path)
        y = np.load(y_path)

    return X, y


def fetch_libsvm(dataset, replace=False, normalize=True, min_nnz=3):
    """
    Download a dataset from LIBSVM website.

    Parameters
    ----------
    dataset : string
        Dataset name. Must be in celer.datasets.libsvm.NAMES.keys()

    replace : bool, default=False
        Whether to force download of dataset if already downloaded.

    normalize : bool
        If True, columns of X are set to unit norm. This may make little sense
        for a sparse matrix since centering is not performed.
        y is centered and set to unit norm if the dataset is a regression one.

    min_nnz: int, default=3
        Columns of X with strictly less than min_nnz non-zero entries are
        discarded.

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
    paths = [CELER_PATH, pjoin(CELER_PATH, 'regression'),
             pjoin(CELER_PATH, 'binary'),
             pjoin(CELER_PATH, 'multilabel'),
             pjoin(CELER_PATH, 'multiclass')]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s" % dataset)
    multilabel = NAMES[dataset].split('/')[0] == 'multilabel'
    is_regression = NAMES[dataset].split('/')[0] == 'regression'

    print("Dataset: %s" % dataset)
    compressed_path = pjoin(CELER_PATH, "%s.bz2" % NAMES[dataset])
    download_libsvm(dataset, compressed_path, replace=replace)

    X, y = get_X_y(dataset, compressed_path, multilabel, replace=replace)

    # preprocessing
    if min_nnz != 0:
        X = X[:, np.diff(X.indptr) >= min_nnz]

    if normalize:
        X = preprocessing.normalize(X, axis=0)
        if is_regression:
            y -= np.mean(y)
            y /= np.std(y)

    return X, y


if __name__ == "__main__":
    for dataset in NAMES:
        fetch_libsvm(dataset, replace=False)
