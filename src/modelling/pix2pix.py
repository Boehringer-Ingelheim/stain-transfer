import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import color

from src.modelling.base_model import BaseModel
from src.modelling.cyclegan import CycleGan
from src.modelling.networks import UnetGenerator


class RGB2L(object):
    """ Converts an RGB PIL.Image to a torch.FloatTensor that contains the L
    channel in the LAB color space."""

    def __call__(self, pic: Image):
        """
        :param pic: Image to be converted to tensor.
        :return L: L channel tensor.
        """

        lab = color.rgb2lab(np.array(pic)).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        L = lab_t[[0], ...] / 50.0 - 1.0

        return L


class Pix2Pix(CycleGan):
    """This class implements the pix2pix grayscale colorization model for
    image-to-image translation inference."""

    def __init__(self, conf):
        """
        Initializes model and create dataset loaders.

        :param conf: See BaseModel.
        """

        BaseModel.__init__(self, conf)
        self.net = UnetGenerator(input_nc=1, output_nc=2, num_downs=8, ngf=64,
                                 use_dropout=False)
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()

    def lab2rgb(self, L: torch.Tensor, AB: torch.Tensor):
        """Convert an Lab tensor image to a RGB numpy output

        :param L: L channel images.
        :param AB: AB channel images.

        :return rgb: RGB output images.
        """

        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = np.transpose(Lab.detach().cpu().float().numpy(),
                           (0, 2, 3, 1)).astype(np.float64)
        rgb = color.lab2rgb(Lab) * 255.0

        return rgb

    def forward(self, t: torch.Tensor):
        """ Run forward pass."""
        return t.to(self.conf.device), self.net(t.to(self.conf.device))

    def post_transform(self, t: tuple):
        """ Post-process transformations to be applied to network outputs."""
        t = self.lab2rgb(t[0], t[1]).astype(np.uint8)

        return t

    def transform(self):
        """ Pre-process transformations to be applied to inputs in data loaders."""
        return [RGB2L()]
