from pathlib import Path
CELER_PATH = str(Path.home()) + '/celer_data/'

from .climate import load_climate
from .libsvm import load_libsvm
from .ml_uci import load_ml_uci

