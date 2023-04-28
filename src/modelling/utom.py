from src.modelling.base_model import BaseModel
from src.modelling.cyclegan import CycleGan
from src.modelling.networks import UnetGenerator, NORM_LAYER


class Utom(CycleGan):
    """This class implements the UTOM model for image-to-image translation inference.
    Utom paper https://www.nature.com/articles/s41377-021-00484-y.pdf."""

    def __init__(self, conf):
        """
        Initializes model and create dataset loaders.

        :param conf: See BaseModel
        """

        BaseModel.__init__(self, conf)
        self.net = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64,
                                 norm_layer=NORM_LAYER, use_dropout=False)
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()
