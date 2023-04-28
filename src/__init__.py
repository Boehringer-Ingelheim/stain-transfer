from pathlib import Path
from shutil import rmtree

from src.data_processing import IMAGE_EXTENSIONS, DATA_DIR
from src.modelling import MODELS, MODEL_INS, MODEL_WEIGHTS, MODEL_DIR, L_MODELS, \
    NL_MODELS, NAMELESS_MODEL, RESULTS_DIR, IMAGE_OUT_DIR

__version__ = "0.0.1"

TISSUES = ["HE", "Masson"]
ASSETS_DIR = Path(__file__).absolute().parents[0] / 'data_processing' / 'assets'
FID_ACT_DIR = RESULTS_DIR / 'fid_activations'
STATS_DIR = RESULTS_DIR / 'statistics'
TMP_DIR = RESULTS_DIR / 'tmp'
rmtree(ASSETS_DIR, ignore_errors=True)
rmtree(TMP_DIR, ignore_errors=True)
ASSETS_DIR.mkdir(parents=True)
FID_ACT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)
