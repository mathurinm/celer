from celer import celer_path
import numpy as np


X = np.ones([3, 4], np.float32)
y = np.ones(3, np.float32)

celer_path(X, y, 'lasso')
