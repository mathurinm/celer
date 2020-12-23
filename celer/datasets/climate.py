# Author : Eugene Ndiaye
#          Mathurin Massias
# BSD License

import os
from os.path import join as pjoin

import numpy as np
import xarray
import download
from scipy.signal import detrend

from celer.datasets import CELER_PATH

FILES = ["air.mon.mean.nc", 'pres.mon.mean.nc', 'pr_wtr.mon.mean.nc',
         "rhum.mon.mean.nc", 'slp.mon.mean.nc', "uwnd.mon.mean.nc",
         "vwnd.mon.mean.nc",
         ]


def _get_data(filename):
    data = xarray.open_dataset(
        pjoin(CELER_PATH, 'climate/surface', filename), decode_times=False)

    n_times = data[list(data.data_vars.keys())[0]].shape[0]

    X = np.array(data[list(data.data_vars.keys())[0]]).reshape(n_times, -1)

    # remove seasonality
    period = 12
    for m in range(period):
        # TODO using sklearn for preprocessing would be an improvement
        X[m::period] -= np.mean(X[m::period], axis=0)[None, :]
        X[m::period] /= np.std(X[m::period], axis=0)[None, :]
        if np.sum(np.isnan(X[m::period])) > 0:
            X[m::period] = np.where(np.isnan(X[m::period]), 0, X[m::period])

    # remove trend
    X = detrend(X, axis=0, type='linear')

    return X


def _download_climate(replace=False):
    prefix = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/"

    for fname in FILES:
        target = pjoin(CELER_PATH, 'climate/surface', fname)
        download.download(prefix + "surface/" + fname, target,
                          replace=replace)


def _target_region(lx, Lx):

    arrays = [_get_data(filename) for filename in FILES]

    n, p = arrays[0].shape
    X = np.zeros((n, 7 * (p - 1)), order='F')

    pos_lx = int((90 - lx) / 2.5)
    pos_Lx = (np.ceil(Lx / 2.5)).astype(int)
    target = pos_lx * 144 + pos_Lx

    begin = 0
    for j in range(p):
        if j == target:
            continue
        X[:, begin:begin + 7] = np.vstack(
            [arr[:, j] for arr in arrays]).T
        begin += 7

    y = arrays[0][:, target].astype(np.float64)

    # np.save(pjoin(path, 'climate_data.npy'), X)
    # np.save(pjoin(path, 'climate_target.npy'), y)

    return X, y


def fetch_climate(replace=False):
    """Get design matrix and observation for the climate dataset.

    Parameters
    ----------
    replace: bool (default=False)
        Whether to redownload the files if already present on disk.

    Returns
    -------
    X: np.array, shape (n_samples, n_features)
        Design matrix.
    y: np.array, shape (n_samples,)
        Observations.
    """
    path = pjoin(CELER_PATH, 'climate')
    if not os.path.exists(path):
        os.mkdir(path)

    _download_climate(replace=replace)
    lx, Lx = 14, 17  # Dakar
    print("Preprocessing and loading target region...")
    X, y = _target_region(lx, Lx)

    return X, y


if __name__ == "__main__":
    X, y = fetch_climate(replace=True)
