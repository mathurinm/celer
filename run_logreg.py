import time
import numpy as np
from sklearn.utils import check_random_state

from celer import LogisticRegression

n_samples, n_features = 10, 100

rng = check_random_state(0)
X = rng.normal(0, 1, (n_samples, n_features))
y = np.sign(rng.normal(0, 1, (n_samples,)))

clf = LogisticRegression(verbose=2).fit(X, y)
