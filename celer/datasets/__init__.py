from pathlib import Path
CELER_PATH = str(Path.home()) + '/celer_data/'  # noqa

from .climate import fetch_climate
from .libsvm import fetch_libsvm
from .ml_uci import fetch_ml_uci
from .simulated import make_correlated_data
