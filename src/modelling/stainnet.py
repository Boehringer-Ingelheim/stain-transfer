import torch

from src.modelling.base_model import BaseModel
from src.modelling.cyclegan import CycleGan
from src.modelling.networks import StainNetNet


class StainNet(CycleGan):
    """
    This class implements the StainNet model for image-to-image translation inference.
    StainNet paper: https://arxiv.org/ftp/arxiv/papers/2012/2012.12535.pdf.
    """

    def __init__(self, conf):
        """
        Initializes model and create dataset loaders.

        :param conf: See BaseModel.
        """

        BaseModel.__init__(self, conf)
        self.net = StainNetNet()
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()

    def load_weights(self):
        """ Load model weights."""
        self.net.load_state_dict(
            torch.load(self.conf.weights, map_location=self.conf.device))
