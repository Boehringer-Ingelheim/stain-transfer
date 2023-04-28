import unittest

from torchvision import transforms

from src import DATA_DIR, RESULTS_DIR, MODEL_WEIGHTS
from src.data_processing.config import ModelConf
from src.modelling.colorstat import ColorStat
from src.modelling.cut import CUT
from src.modelling.cyclegan import CycleGan
from src.modelling.macenko import Macenko
from src.modelling.munit import Munit
from src.modelling.pix2pix import Pix2Pix
from src.modelling.unit import Unit
from src.modelling.vahadane import Vahadane


class Test_TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf = {"device": "cpu", "num_workers": 4, "batch_size": 1,
                "data_path": DATA_DIR / 'processed' / 'HE',
                "results_path": RESULTS_DIR, 'add_suffix': False,
                'center_crop': False, 'a2b': True}
        cls.target_dir = DATA_DIR / 'processed' / 'masson_trichrome'
        cls.target_samples = 2
        cls.conf = ModelConf(**conf)

    def basic_model_test(self, model):
        model.eval()
        image_tensor, image_path = next(iter(model.dataloader))
        if model.__class__.__name__ in [Munit.__name__, ColorStat.__name__,
                                        Macenko.__name__, Vahadane.__name__]:
            # These models perform different fit on each forward, so repeat
            # input image and do only one forward.
            tensor = image_tensor.repeat(2, 1, 1, 1)
            output = model.post_transform(model.forward(tensor))
            output_tensor = transforms.Compose(model.transform())(output[0])
            output_tensor2 = transforms.Compose(model.transform())(output[1])
            if model.__class__.__name__ == Vahadane.__name__:
                # Vahadane does learning on inference when getting stain matrix
                # for source image.
                output_tensor2 = output_tensor
        else:
            output_tensor = transforms.Compose(model.transform())(
                model.post_transform(model.forward(image_tensor)).squeeze())
            output_tensor2 = transforms.Compose(model.transform())(
                model.post_transform(model.forward(image_tensor)).squeeze())

        self.assertEqual(image_tensor.squeeze(dim=0).numpy().shape,
                         output_tensor.shape)
        self.assertEqual(output_tensor.numpy().tolist(),
                         output_tensor2.numpy().tolist())

    def test_CycleGan(self):
        Test_TestModels.conf.weights = MODEL_WEIGHTS['a2b']['cyclegan']
        model = CycleGan(Test_TestModels.conf)
        self.basic_model_test(model)

    def test_CUT(self):
        Test_TestModels.conf.weights = MODEL_WEIGHTS['a2b']['cut']
        model = CUT(Test_TestModels.conf)
        self.basic_model_test(model)

    def test_Pix2Pix(self):
        Test_TestModels.conf.weights = MODEL_WEIGHTS['a2b']['pix2pix']
        model = Pix2Pix(Test_TestModels.conf)
        self.basic_model_test(model)

    def slow_test_Unit(self):
        Test_TestModels.conf.weights = MODEL_WEIGHTS['a2b']['unit']
        model = Unit(Test_TestModels.conf)
        self.basic_model_test(model)

    def test_Munit(self):
        Test_TestModels.conf.weights = MODEL_WEIGHTS['a2b']['munit']
        Test_TestModels.conf.target_path = Test_TestModels.target_dir
        model = Munit(Test_TestModels.conf)
        self.basic_model_test(model)

    def test_Macenko(self):
        Test_TestModels.conf.target_path = Test_TestModels.target_dir
        Test_TestModels.conf.target_samples = Test_TestModels.target_samples
        model = Macenko(Test_TestModels.conf)
        self.basic_model_test(model)

    def test_ColorStat(self):
        Test_TestModels.conf.target_path = Test_TestModels.target_dir
        Test_TestModels.conf.target_samples = Test_TestModels.target_samples
        model = ColorStat(Test_TestModels.conf)
        self.basic_model_test(model)

    def test_Vahadane(self):
        Test_TestModels.conf.target_path = Test_TestModels.target_dir
        Test_TestModels.conf.target_samples = Test_TestModels.target_samples
        model = Vahadane(Test_TestModels.conf)
        self.basic_model_test(model)
