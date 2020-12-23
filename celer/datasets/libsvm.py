# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import warnings

import libsvmdata


def fetch_libsvm(dataset, replace=False, normalize=True, min_nnz=3):
    """
    This function is deprecated, we now rely on the libsvmdata package.

    Parameters
    ----------
    dataset: string
        Name of the dataset.
    replace: bool
        Whether to redownload the data.
    normalize: bool
        Whether to divide the columns by their norm.
    min_nnz: int
        Columns with strictly less than `nnz` non-zero entries are discarded.
    """
    warnings.simplefilter("always", FutureWarning)
    warnings.warn("celer.datasets.fetch_libsvm is deprecated and will be "
                  "removed in version 0.6. Use the lightweight "
                  "libsvmadata package instead.", FutureWarning)
    return libsvmdata.fetch_libsvm(dataset, replace=replace,
                                   normalize=normalize, min_nnz=min_nnz)


if __name__ == "__main__":
    for dataset in libsvmdata.datasets.NAMES:
        fetch_libsvm(dataset, replace=False)
