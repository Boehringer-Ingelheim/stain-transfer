from src.modelling.macenko import Macenko, MacenkoAverageTensor


class Vahadane(Macenko):
    """This class implements the Vahadane stain normalization model."""
    normalization = "vahadane"

    def __init__(self, conf):
        """
        Initializes Vahadane model.

        :param conf: See BaseModel.
        """

        Macenko.__init__(self, conf)


class VahadaneAverageTensor(MacenkoAverageTensor, Vahadane):
    """VahadaneAverageTensor"""

    def __init__(self, conf: dict):
        conf.target_samples = -1
        conf.target_path = conf.data_path
        Vahadane.__init__(self, conf)
        self.set_out_tensor_file_name()
