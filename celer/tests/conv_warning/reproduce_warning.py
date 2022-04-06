import numpy as np
from celer import LogisticRegression

from celer.tests.conv_warning.logs.dumped_data import DICT_DATA

data = DICT_DATA['check_fit_idempotent']
X = data["X"]
y = data["y"]
C = data["C"]

clf = LogisticRegression(C=C)
clf.fit(X, y)
