from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import IMAGE_OUT_DIR
from src.data_processing.metrics import SSIM
from src.modelling.pix2pix import Pix2Pix
from src.modelling import NL_MODELS


def get_rotated_model(model):
    class RotatedModel(model):
        """
        This class applies a rotation transformation to input tensors. Each input
        image suffers a 90°, 180° and 270° rotation before being passed to the
        network, therefore batch size is expanded from BS to BS*4.
        """

        def __init__(self, conf):
            """
            Initializer. Class name is changed to dynamic model name and
            image output directory is also changed.
            :param conf: See BaseModel.
            """

            self.__class__.__name__ = model.__name__
            model.__init__(self, conf)
            now = datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.image_outs = IMAGE_OUT_DIR / "Rotated" / self.__class__.__name__ / now
            self.rotations = range(1, 4)
            self.anti_rotations = self.rotations[::-1]

        def apply_rotation(self, t: torch.tensor, k: int):
            """Applies k 90° rotations to input tensor images."""
            dims = (2, 3)
            if self.__class__.__name__ in NL_MODELS:
                # Non learning models use channel at the end and not after batch.
                dims = (1, 2)
            return torch.rot90(t, k, dims)

        def rotated_forward(self, t: torch.Tensor):
            """
            Runs rotated forward pass:
            1) Rotations to input tensors are applied and appended to this input tensor.
            2) Model forward is called.
            3) Respective anti rotations are then applied to all rotated tensors.

            The original output tensor and the anti rotated tensors are returned as a tuple.
            """

            originals = len(t)
            cat = torch.cat(
                [t] + [self.apply_rotation(t, i) for i in self.rotations])
            output = self.forward(cat)

            if Pix2Pix.__name__ in self.__class__.__name__:
                # Pix2Pix forward returns tuple of tensors.
                t = (output[0][:originals], output[1][:originals])
            else:
                t = output[:originals]

            anti_rotated = []
            for i, x in enumerate(self.anti_rotations):
                start_idx = originals * (i + 1)
                end_idx = start_idx + originals
                if Pix2Pix.__name__ in self.__class__.__name__:
                    # Pix2Pix forward returns tuple of tensors.
                    rot_0 = output[0][start_idx:end_idx]
                    rot_1 = output[1][start_idx:end_idx]
                    anti_rotated.append((self.apply_rotation(rot_0, x),
                                         self.apply_rotation(rot_1, x)))
                else:
                    rot = output[start_idx:end_idx]
                    anti_rotated.append(self.apply_rotation(rot, x))

            return t, anti_rotated

        def predict(self):
            """
            Forward function used in test time. This function sets the model
            to eval mode and wraps <rotated_forward> function in no_grad()
            so we don't save intermediate steps for backprop. It also calls
            <save_outputs> to store results and gets average SSIM between
            output images and rotated versions of these.
            """

            self.image_outs.mkdir(parents=True, exist_ok=True)
            self.eval()
            ssims = defaultdict(list)
            with torch.no_grad():
                for imgs, paths in tqdm(self.dataloader):
                    t, rotated = self.rotated_forward(imgs)
                    t = self.post_transform(t)
                    self.save_outputs(t, paths)
                    for i, rot in enumerate(rotated):
                        rot = self.post_transform(rot)
                        angle = 90 * self.rotations[i]
                        rot_path = [Path(x).parents[
                                        0] / f"{Path(x).stem}_{angle}{Path(x).suffix}"
                                    for x in paths]
                        self.save_outputs(rot, rot_path)
                        ssims[angle] += [SSIM(x1, x2) for x1, x2 in zip(rot, t)]
            av_ssims = {k: np.array(v).mean() for k, v in ssims.items()}

            return av_ssims

    return RotatedModel


def rotate_model(model):
    return get_rotated_model(model)
