from pathlib import Path

# Defined here to avoid partially initialized module import errors.
RESULTS_DIR = Path(__file__).absolute().parents[2] / 'results'
IMAGE_OUT_DIR = RESULTS_DIR / 'images'
AV_TENSOR_OUT_DIR = RESULTS_DIR / 'av_tensors'
NAMELESS_MODEL = ""

from .colorstat import ColorStat, ColorStatAverageTensor
from .cut import CUT
from .cyclegan import CycleGan
from .drit import Drit, DritAverageTensor
from .macenko import Macenko, MacenkoAverageTensor
from .pix2pix import Pix2Pix
from .staingan import StainGan
from .stainnet import StainNet
from .unit import Unit
from .munit import Munit, MunitAverageTensor
from .utom import Utom
from .vahadane import Vahadane, VahadaneAverageTensor

MODEL_DIR = Path(__file__).absolute().parents[2] / 'models'
NL_MODELS = [ColorStat.__name__, ColorStatAverageTensor.__name__,
             Macenko.__name__, MacenkoAverageTensor.__name__,
             Vahadane.__name__, VahadaneAverageTensor.__name__]
L_MODELS = [CUT.__name__,
            CycleGan.__name__,
            Drit.__name__,
            Munit.__name__,
            MunitAverageTensor.__name__,
            Pix2Pix.__name__,
            StainGan.__name__,
            StainNet.__name__,
            Unit.__name__,
            Utom.__name__]
MODELS = NL_MODELS + L_MODELS

MODEL_INS = {"cut": CUT,
             "cyclegan": CycleGan,
             "drit": Drit,
             "dritaveragetensor": DritAverageTensor,
             "munit": Munit,
             "munitaveragetensor": MunitAverageTensor,
             "pix2pix": Pix2Pix,
             "staingan": StainGan,
             "stainnet": StainNet,
             "unit": Unit,
             "utom": Utom,
             "colorstat": ColorStat,
             "colorstataveragetensor": ColorStatAverageTensor,
             "macenko": Macenko,
             "macenkoaveragetensor": MacenkoAverageTensor,
             "vahadane": Vahadane,
             "vahadaneaveragetensor": VahadaneAverageTensor}

HE2MT = {"cut": str(MODEL_DIR / 'cut_he2mt.pth'),
         "cyclegan": str(MODEL_DIR / 'cyclegan_he2mt.pth'),
         "drit": str(MODEL_DIR / 'drit.pth'),
         "dritaveragetensor": str(MODEL_DIR / 'drit.pth'),
         "munit": str(MODEL_DIR / 'munit.pth'),
         "munitaveragetensor": str(MODEL_DIR / 'munit.pth'),
         "pix2pix": str(MODEL_DIR / 'pix2pix_he2mt.pth'),
         "staingan": str(MODEL_DIR / 'staingan_he2mt.pth'),
         "stainnet": str(MODEL_DIR / 'stainnet_he2mt.pth'),
         "unit": str(MODEL_DIR / 'unit.pth'),
         "utom": str(MODEL_DIR / 'utom_he2mt.pth'),
         "colorstat": str(MODEL_DIR / 'colorstat_he2mt.pth'),
         "colorstataveragetensor": None,
         "macenko": str(MODEL_DIR / 'macenko_he2mt.pth'),
         "macenkoaveragetensor": None,
         "vahadane": str(MODEL_DIR / 'vahadane_he2mt.pth'),
         "vahadaneaveragetensor": None}
MT2HE = {k: v.replace('_he2mt', '_mt2he') if v else v for k, v in HE2MT.items()}
MODEL_WEIGHTS = {'a2b': HE2MT, 'b2a': MT2HE}
