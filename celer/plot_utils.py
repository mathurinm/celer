# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


def configure_plt():
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_context("poster")
    sns.set_style("ticks")


def plot_path_hist(results, labels, tols, figsize, ylim=None):
    configure_plt()
    sns.set_palette('colorblind')
    n_competitors = len(results)
    fig, ax = plt.subplots(figsize=figsize)
    width = 1. / (n_competitors + 1)
    ind = np.arange(len(tols))
    b = (1 - n_competitors) / 2.
    for i in range(n_competitors):
        plt.bar(ind + (i + b) * width, results[i], width,
                label=labels[i])
    ax.set_ylabel('path computation time (s)')
    ax.set_xticks(ind + width / 2)
    plt.xticks(range(len(tols)), ["%.0e" % tol for tol in tols])
    if ylim is not None:
        plt.ylim(ylim)

    ax.set_xlabel(r"$\epsilon$")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=False)
    return fig
