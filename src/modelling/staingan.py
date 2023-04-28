from src.modelling.base_model import BaseModel
from src.modelling.cyclegan import CycleGan
from src.modelling.networks import ResnetGenerator, NORM_LAYER


class StainGan(CycleGan):
    """
    This class implements the StainGan model for image-to-image translation inference.
    StainGan paper: https://arxiv.org/pdf/1804.01601.pdf.
    """

    def __init__(self, conf):
        """
        Initializes model and create dataset loaders.

        :param conf: See BaseModel.
        """

        BaseModel.__init__(self, conf)
        self.net = ResnetGenerator(3, 3, 64, use_dropout=True, n_blocks=9,
                                   norm_layer=NORM_LAYER)
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()
