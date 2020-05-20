import os

import numpy as np
from sklearn.linear_model import MultiTaskLassoCV
from celer import MultiTaskLassoCV as MTLCV_celer
from numpy.linalg import norm

cur_dir = os.getcwd().split('/')[-1]

if cur_dir == 'chevalier_nguyen':
    data_path = 'hidimstat/examples/chevalier/mne/data_examples/'
elif cur_dir in ['jchevalier', 'jerome']:
    data_path = 'chevalier/hidimstat/examples/chevalier/mne/data_examples/'
else:
    data_path = './'

celer = True
adjust_celer_tol = True
tol = 1e-4
max_iter = 1e4
n_jobs = 1

X = np.load(data_path + 'X_MTL.npy')
Y = np.load(data_path + 'Y_MTL.npy')

Y = np.asfortranarray(Y)
# X /= norm(X, axis=0)[None, :]

n_samples, n_features = X.shape
n_targets = Y.shape[1]

if int(max_iter / 5) <= n_features:
    max_iter = n_features * 5

if adjust_celer_tol:
    tol = tol * 1e-2
clf = \
    MTLCV_celer(tol=tol, max_iter=5, cv=2, eps=1e-1,
                n_jobs=n_jobs, fit_intercept=False,
                normalize=False, verbose=2, n_alphas=2)
clf.fit(X, Y)
