import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from celer.dropin_sklearn import LogisticRegression
from celer.tests.conv_warning.logs.dumped_data import DICT_DATA

import pickle


filename = './logs/gaps_for_mul_alphas.pkl'

# to be excuted only once
# before, uncomment results in homotopy.py


def simulate_LogReg_mul_alphas():
    check_props = DICT_DATA['check_fit_check_is_fitted']
    enc = LabelEncoder()

    X = check_props['X']

    y = check_props['y']
    y_ind = enc.fit_transform(y)

    C = check_props['C']
    alpha_max = norm(X.T.dot(y_ind), ord=np.inf)
    C_max = 1 / alpha_max
    tol = 1e-14

    arr_C = np.linspace(C_max, C, num=5)[1:]  # skip first value

    dict_gaps = {}
    for current_C in arr_C:
        clf = LogisticRegression(C=C, tol=tol, max_iter=100, verbose=0)
        _, gaps = clf.path(
            X, 2 * y_ind - 1, np.array([current_C]), solver="celer-pn")

        current_alpha = 1 / current_C
        plot_name = f'{current_alpha/alpha_max:.2e}'
        dict_gaps[plot_name] = gaps
    # save logs
    with open(filename, 'wb') as f:
        pickle.dump(dict_gaps, f)

    return dict_gaps()


# load
try:
    with open(filename, 'rb') as f:
        dict_gaps = pickle.load(f)
except FileNotFoundError:
    dict_gaps = simulate_LogReg_mul_alphas()

# plot
fig, ax = plt.subplots()

for plot_name, gaps in dict_gaps.items():
    ax.semilogy(gaps, label=plot_name, marker='.')

# set layout
plt.title("LogReg for different alpha")
ax.set_xlabel("iterations")
ax.set_ylabel("dual gap")

plt.grid()
plt.legend(title="Fraction of alpha_max")

plt.show()
