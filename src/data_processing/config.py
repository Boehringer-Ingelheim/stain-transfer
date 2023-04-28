from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from src import MODEL_INS, L_MODELS, MODEL_WEIGHTS, RESULTS_DIR
from src.data_processing import DATA_DIR
from src.utils.utils import check_exists, check_path


@dataclass
class Models:
    names: Sequence[str] = field(default_factory=list)
    weights: Optional[Sequence[str]] = None
    a2b: bool = True
    indent: int = 6

    def __post_init__(self):
        self.names = [x.lower() for x in self.names]
        self.a2b = 'a2b' if self.a2b else 'b2a'
        if self.weights is None:
            self.weights = self._default_weights
        elif len(self.names) != len(self.weights):
            msg = f"Provided list of model names {self.names} and weights {self.weights} do not match. Reverting to default weights."
            print(f"WARNING: {msg}")
            self.weights = self._default_weights
        self.models = [MODEL_INS[x] for x in self.names]
        for model, weight in zip(self.models, self.weights):
            if model.__name__ in L_MODELS:
                check_exists(weight)

    def __str__(self):
        ret = ""
        for n, w in zip(self.names, self.weights):
            ret += "\n".ljust(self.indent) + f"{n}: {w}"

        return ret

    @property
    def _default_weights(self):
        return [MODEL_WEIGHTS[self.a2b][x] for x in self.names]


@dataclass
class ModelConf:
    data_path: Path = DATA_DIR / 'processed' / 'HE'
    results_path: Path = RESULTS_DIR
    batch_size: int = 1
    device: int = -1
    add_suffix: bool = False
    a2b: bool = True
    center_crop: Optional[int] = None
    target_path: Optional[str] = None
    target_samples: Optional[int] = None
    target_tensor: Optional[str] = None
    num_workers: int = 4
    rotate: bool = False

    def __post_init__(self):
        check_path(self.data_path)
        if self.target_path is not None: check_path(self.target_path)
        if self.target_tensor is not None: check_exists(self.target_tensor)


@dataclass
class GenerateConf(ModelConf):
    models: Models = None
    weights: str = None  # Used only for retrieving current weights from list of models

    def __post_init__(self):
        super().__post_init__()
        self.device = 'cpu' if self.device < 0 else f"cuda:{self.device}"
        if self.models is None:
            raise ValueError("models field not provided.")
        if isinstance(self.models, dict):
            self.models = Models(**self.models, a2b=self.a2b)

    def __str__(self):
        ret = "Fake generation:\n"
        ind = '  '
        for x in self.__dataclass_fields__:
            if x == 'weights': continue
            value = getattr(self, x)
            ret += f"{ind}{x}: {value}\n"

        return ret


@dataclass
class MetricsConf:
    classic_metrics: Sequence[str] = ('ssim',)
    source: Path = DATA_DIR / 'processed' / 'HE'  # source images path
    fake: Path = RESULTS_DIR / 'images' / 'fakes'  # fake images path
    results_path: Path = RESULTS_DIR  # results path
    center_crop: Optional[int] = None  # size of center crop transformation
    target: Optional[Path] = None  # target images path

    def __post_init__(self):
        self.classic_metrics = [x.lower() for x in self.classic_metrics]
