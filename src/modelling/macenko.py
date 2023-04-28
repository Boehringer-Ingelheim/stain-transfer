from pathlib import Path

import numpy as np
import staintools
import torch
from PIL import Image
from tqdm import tqdm

from src.modelling.colorstat import ColorStat


class LuminosityStandardizer(object):
    """ Applies LuminosityStandardizer to an RGB PIL.Image"""

    def __call__(self, pic: Image):
        """
        :param pic: Image to be converted to tensor.
        """

        return torch.tensor(
            staintools.LuminosityStandardizer.standardize(np.array(pic)))


class Macenko(ColorStat):
    """This class implements the Macenko stain normalization model."""
    normalization = "macenko"
    MODE = {
        1: "computing stain matrices and maxC from configuration target_path",
        2: "using stain matrices and maxC from configuration target_tensor",
        3: "using stain matrices and maxC from configuration weights"}

    def __init__(self, conf):
        """
        Initializes Macenko model.

        :param conf: See BaseModel
        """

        conf.device = 'cpu'
        self.normalizer = staintools.StainNormalizer(
            self.__class__.normalization)
        ColorStat.__init__(self, conf)

    def full_target_train(self):
        self.compute_full_stain_matrix()

    def load_average(self, average_tensor_path: str):
        s = torch.load(average_tensor_path)
        self.normalizer.stain_matrix_target = s['stain_matrix']
        self.normalizer.maxC_target = s['maxC']

    def accumulate_stain_matrix(self, targets: list, filenames: list,
                                acc_stain_matrix_target: np.ndarray,
                                acc_maxC_target: np.ndarray, total: int):
        for target, fname in zip(targets, filenames):
            try:
                self.normalizer.fit(np.array(target))
            except AssertionError as E:
                print(f"Skipping {fname} from computation due to {E}")
                continue
            total += 1
            acc_stain_matrix_target += self.normalizer.stain_matrix_target
            acc_maxC_target += self.normalizer.maxC_target

        return acc_stain_matrix_target, acc_maxC_target, total

    def compute_full_stain_matrix(self):
        stain_matrix_target = np.zeros(shape=(2, 3))
        maxC_target = np.zeros(shape=(1, 2))
        total = 0
        print(f'{self.__class__.__name__} computing average stain matrix '
              f'for {self.conf.target_path}.')
        progress_bar = tqdm(total=len(self.target_iterator))
        while True:
            try:
                target, f_name = next(self.target_iterator)
                progress_bar.update(1)
                stain_matrix_target, maxC_target, total = self.accumulate_stain_matrix(
                    target, f_name, stain_matrix_target, maxC_target, total)
            except StopIteration:
                del self.target_iterator
                break
        progress_bar.close()
        self.normalizer.stain_matrix_target = stain_matrix_target / total
        self.normalizer.maxC_target = maxC_target / total

    def transform(self):
        """Pre-process transformations applied to inputs in data loaders."""
        return [LuminosityStandardizer()]

    def post_transform(self, x: torch.Tensor):
        """ Post-process transformations applied to outputs."""
        return np.array([img.numpy().astype(np.uint8) for img in x])

    def forward(self, x: torch.Tensor):
        """ Run forward pass."""
        transformed = []
        if self.conf.target_samples != -1 and self.conf.target_path is not None:
            targets, f_names = self.get_targets()
            stain_matrix_target = np.zeros(shape=(2, 3))
            maxC_target = np.zeros(shape=(1, 2))
            total = 0
            stain_matrix_target, maxC_target, total = self.accumulate_stain_matrix(
                targets, f_names, stain_matrix_target, maxC_target, total)
            self.normalizer.stain_matrix_target = stain_matrix_target / total
            self.normalizer.maxC_target = maxC_target / total

        for img in x:
            transformed.append(self.normalizer.transform(np.array(img)))

        return torch.tensor(transformed)


class MacenkoAverageTensor(Macenko):
    """MacenkoAverageTensor"""

    def __init__(self, conf: dict):
        conf.target_samples = -1
        conf.target_path = conf.data_path
        Macenko.__init__(self, conf)
        self.set_out_tensor_file_name()

    def set_out_tensor_file_name(self):
        dp = Path(self.conf.data_path)
        filename = '_'.join(dp.parts[1:] if dp.is_absolute() else dp.parts)
        self.filename = (self.image_outs / filename).with_suffix('.pth')

    def predict(self):
        self.image_outs.mkdir(parents=True, exist_ok=True)
        torch.save({"stain_matrix": self.normalizer.stain_matrix_target,
                    "maxC": self.normalizer.maxC_target}, self.filename)
        print(f"Saved stain_matrix and maxC to {self.filename}")
        print(f"Computed:\nstain_matrix: {self.normalizer.stain_matrix_target}"
              f"\nmaxC: {self.normalizer.maxC_target}")
