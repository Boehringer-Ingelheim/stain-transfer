import unittest

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src import DATA_DIR
from src.data_processing import metrics
from src.data_processing.datasets import get_img_files, ImagePathDataset


class Test_TestImagePathDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.he_path = DATA_DIR / 'processed' / 'HE'
        cls.mt_path = DATA_DIR / 'processed' / 'masson_trichrome'
        he_dataset = ImagePathDataset(cls.he_path, transforms=ToTensor())
        he_dataloader = DataLoader(he_dataset, batch_size=2)
        cls.samples = next(iter(he_dataloader))

    def test_get_image_files(self):
        self.assertGreaterEqual(
            len(get_img_files(Test_TestImagePathDataset.he_path)), 4)
        self.assertGreaterEqual(
            len(get_img_files(Test_TestImagePathDataset.mt_path)), 4)

    def test_batch_dimensions(self):
        image_tensor, image_paths = Test_TestImagePathDataset.samples
        self.assertEqual(len(image_paths), 2)  # 2 paths
        self.assertEqual(len(image_tensor.shape), 4)  # 4 dim
        self.assertEqual(image_tensor.shape[0], 2)  # 2 samples
        self.assertEqual(image_tensor.shape[1], 3)  # 3 channels

    def test_transforms(self):
        image_tensor, image_paths = Test_TestImagePathDataset.samples
        for i in range(len(image_tensor)):
            self.assertTrue((image_tensor[i] == ToTensor()(
                Image.open(image_paths[i]))).all())


class Test_TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _r = range(3)
        cls.img1 = np.random.randint(low=0, high=256,
                                     size=(256, 256, 3)).astype(np.uint8)
        cls.img2 = np.random.randint(low=0, high=50, size=(256, 256, 3)).astype(
            np.uint8)
        cls.zeros = np.zeros((256, 256, 3)).astype(np.uint8)
        cls.img1_hist = metrics.get_hist(cls.img1)
        cls.img2_hist = metrics.get_hist(cls.img2)
        cls.zeros_hist = metrics.get_hist(cls.zeros)
        cls.white_hist = metrics.get_hist(255 + cls.zeros)

    def test_ssim(self):
        s1 = metrics.SSIM(Test_TestMetrics.img1, Test_TestMetrics.zeros)
        s2 = metrics.SSIM(Test_TestMetrics.img2, Test_TestMetrics.zeros)
        self.assertAlmostEqual(
            metrics.SSIM(255 + Test_TestMetrics.zeros, Test_TestMetrics.zeros),
            0, places=3)
        self.assertEqual(
            metrics.SSIM(Test_TestMetrics.img1, Test_TestMetrics.img1), 1)
        self.assertLess(s1, s2)

    def test_fsim(self):
        self.assertTrue(np.isnan(
            metrics.FSIM(Test_TestMetrics.zeros, Test_TestMetrics.img1)))
        self.assertTrue(np.isnan(
            metrics.FSIM(Test_TestMetrics.zeros, Test_TestMetrics.img1)))
        self.assertEqual(
            metrics.FSIM(Test_TestMetrics.img1, Test_TestMetrics.img1), 1)
        self.assertEqual(
            metrics.FSIM(Test_TestMetrics.img2, Test_TestMetrics.img2), 1)

    def test_av_wasserstein(self):
        w1 = metrics.av_wasserstein(Test_TestMetrics.img1_hist,
                                    Test_TestMetrics.zeros_hist)
        w2 = metrics.av_wasserstein(Test_TestMetrics.img2_hist,
                                    Test_TestMetrics.zeros_hist)
        self.assertEqual(metrics.av_wasserstein(Test_TestMetrics.img1_hist,
                                                Test_TestMetrics.img1_hist), 0)
        self.assertGreater(w1, w2)

    def test_Kullback_Leibler(self):
        k1 = metrics.Kullback_Leibler(Test_TestMetrics.zeros_hist,
                                      Test_TestMetrics.img1_hist)
        k2 = metrics.Kullback_Leibler(Test_TestMetrics.zeros_hist,
                                      Test_TestMetrics.img2_hist)
        self.assertEqual(metrics.Kullback_Leibler(Test_TestMetrics.img1_hist,
                                                  Test_TestMetrics.img1_hist),
                         0)
        self.assertGreater(k1, k2)

    def test_hellinger(self):
        h1 = metrics.hellinger(Test_TestMetrics.zeros_hist,
                               Test_TestMetrics.img1_hist)
        h2 = metrics.hellinger(Test_TestMetrics.zeros_hist,
                               Test_TestMetrics.img2_hist)
        self.assertEqual(metrics.hellinger(Test_TestMetrics.img1_hist,
                                           Test_TestMetrics.img1_hist), 0)
        self.assertEqual(metrics.hellinger(Test_TestMetrics.white_hist,
                                           Test_TestMetrics.zeros_hist), 1)
        self.assertGreater(h1, h2)

    def test_Bhattacharyya(self):
        h1 = metrics.Bhattacharyya(Test_TestMetrics.zeros_hist,
                                   Test_TestMetrics.img1_hist)
        h2 = metrics.Bhattacharyya(Test_TestMetrics.zeros_hist,
                                   Test_TestMetrics.img2_hist)
        self.assertEqual(metrics.Bhattacharyya(Test_TestMetrics.img1_hist,
                                               Test_TestMetrics.img1_hist), 1)
        self.assertEqual(metrics.Bhattacharyya(Test_TestMetrics.white_hist,
                                               Test_TestMetrics.zeros_hist), 0)
        self.assertLess(h1, h2)
