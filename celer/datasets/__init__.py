from pathlib import Path
CELER_PATH = str(Path.home()) + '/celer_data/'

from .climate import fetch_climate
from .libsvm import fetch_libsvm
from .ml_uci import fetch_ml_uci

