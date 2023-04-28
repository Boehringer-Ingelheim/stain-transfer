from src.modelling.cyclegan import CycleGan


class CUT(CycleGan):
    """
    This class implements the CUT model generator, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation.
    The generator is very similar to that of CycleGan.
    """

    no_antialias = False
    no_antialias_up = False

    def __init__(self, conf):
        """
        Initializes CUT model.

        :param conf: See BaseModel.
        """

        CycleGan.__init__(self, conf)
