from celer.dropin_sklearn import LogisticRegression
from celer.tests.conv_warning.dumped_data import DICT_DATA


for check_name, check_props in DICT_DATA.items():
    X = check_props['X']
    y = check_props['y']
    C = check_props['C']
    tol = check_props['tol']

    clf = LogisticRegression(C=C, tol=tol, verbose=0)
    clf.fit(X, y)
