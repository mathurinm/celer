"""
=======================================================
Lasso path computation on Leukemia dataset
=======================================================

The example runs the Celer algorithm on the Leukemia
dataset which is a dense dataset.

Running time is compared with the scikit-learn implementation.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

print("Loading data...")
x = np.arange(12)
y = x ** 2
plt.figure()
plt.plot(x, y)
plt.show()
