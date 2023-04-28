from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import phasepack.phasecong as pc
from PIL import Image
from cleanfid import fid
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from skimage import color
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import CenterCrop

import src.utils.utils as ut
from src import STATS_DIR
from src.data_processing.config import MetricsConf
from src.data_processing.datasets import get_img_files


def SSIM(img1: np.ndarray, img2: np.ndarray) -> np.float64:
    """
    Calcualte SSIM between two images. Convert to grayscale before.

    :param img1: Image 1 in RGB color space
    :param img2: Image 2 in RGB color space
    :return: Computed SSIM
    """

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(img1, img2, multichannel=False)


def _similarity_measure(x: np.ndarray, y: np.ndarray,
                        constant: float) -> np.ndarray:
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def _gradient_magnitude(img: np.ndarray, img_depth: int) -> np.ndarray:
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def FSIM(org_img: np.ndarray, pred_img: np.ndarray, T1=0.85,
         T2=160) -> np.float64:
    alpha, beta = 1, 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(3):
        # Calculate the PC for original and predicted images
        # pc is rally really slow. Vectorizing the rest doesn't speed up.
        pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2,
                      sigmaOnf=0.5978)[4]
        pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2,
                      sigmaOnf=0.5978)[4]

        # pc1_2dim and pc2_2dim are tuples with length 7, we only need the 4th
        # element which is the PC. The PC itself is a list with size 6 (number
        # of orientation). Therefore, we need to calculate the sum of all these
        # 6 arrays.
        pc1_2dim_sum = np.array(pc1_2dim).sum(axis=0)
        pc2_2dim_sum = np.array(pc2_2dim).sum(axis=0)

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def av_wasserstein(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    return np.mean([wasserstein_distance(hist1[i], hist2[i]) for i in range(3)])


def Kullback_Leibler(true_hist: np.ndarray,
                     other_hist: np.ndarray) -> np.float64:
    ohc = other_hist.copy()
    ohc[other_hist == 0] += 1e-10

    return np.mean([entropy(pk=true_hist[i], qk=ohc[i]) for i in range(3)])


def hellinger(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    return np.mean(
        [(euclidean(hist1[i] ** 0.5, hist2[i] ** 0.5)) / np.sqrt(2) for i in
         range(3)])


def Bhattacharyya(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    return np.mean([((hist1[i] * hist2[i]) ** 0.5).sum() for i in range(3)])


def get_classic_metrics(metrics: Sequence[str], source: np.ndarray,
                        fake: np.ndarray, fake_h: np.ndarray = None,
                        target_h: np.ndarray = None) -> dict:
    metric_dict = {}
    for name, metric in zip(('SSIM', 'FSIM'), (SSIM, FSIM)):
        if name.lower() in metrics:
            metric_dict[name] = metric(source, fake)
    if fake_h is not None and target_h is not None:
        n = ('Wasserstein', 'Kullback_Leibler', 'Hellinger', 'Bhattacharyya')
        m = (av_wasserstein, Kullback_Leibler, hellinger, Bhattacharyya)
        for name, metric in zip(n, m):
            if name.lower() in metrics:
                metric_dict[name] = metric(target_h, fake_h)

    return metric_dict


def get_hist(img: np.ndarray) -> np.ndarray:
    """
    Get the histogram of an image.

    :param img: image whose histogram is to be computed.
    :return: image histogram.
    """
    hist = [np.histogram(img[..., j].flatten(), bins=256, range=[0, 256],
                         density=True)[0] for j in range(3)]

    return np.array(hist)


def triplet2file_img_hist(triplet: dict, image_type: str, image_file: str,
                          model: str = None, center_crop_size: int = 0):
    """
    Gets the filename, image and its histogram from a triplet dictionary.

    :param triplet: triplet dictionary of source, fakes and target images.
    :param image_type: one of ['source','fakes','target']
    :param image_file: image file common stem.
    :param model: model name, only valid for fake images.
    :param center_crop_size: size of center crop transformation.
    """

    try:
        if 'fake' in image_type and model is not None:
            file = Path(triplet[image_type][model][image_file])
        else:
            file = Path(triplet[image_type][image_file])
        if center_crop_size:
            img = np.array(
                CenterCrop(center_crop_size)(Image.open(file).convert('RGB')))
        else:
            img = np.array(Image.open(file).convert('RGB'))
        hist = get_hist(img)
    except KeyError:
        file, img, hist = None, None, None

    return file, img, hist


def generate_classic_metrics(metrics_conf: MetricsConf) -> pd.DataFrame:
    """
    Computes classic metrics for images in the <source>, <fake> and <target>
    directories. The <fake> directory can contain fakes from multiple models.

    :param metrics_conf: MetricsConf object
    :return: Dataframe with metrics.
    """

    if metrics_conf.results_path is None:
        results_path = STATS_DIR
    else:
        ut.check_path(metrics_conf.results_path)
        results_path = Path(metrics_conf.results_path) / 'statistics'
        results_path.mkdir(parents=True, exist_ok=True)

    images = ut.get_triplets(metrics_conf.source, metrics_conf.fake,
                             metrics_conf.target)
    metrics_df = pd.DataFrame()
    for m in ut.MODELS:
        metrics = {}
        if metrics_conf.target is None:
            matches = ut.intersect_lists(images['source'].keys(),
                                         images['fakes'][m].keys())
        else:
            matches = ut.intersect_lists(images['source'].keys(),
                                         images['fakes'][m].keys(),
                                         images['target'].keys())
        for x in matches:
            s, s_img, _ = triplet2file_img_hist(images, 'source', x,
                                                center_crop_size=metrics_conf.center_crop)
            f, f_img, f_hist = triplet2file_img_hist(images, 'fakes', x, m)
            _, t_img, t_hist = triplet2file_img_hist(images, 'target', x)
            if s_img.shape == f_img.shape or True:
                metrics[s.name] = get_classic_metrics(
                    metrics_conf.classic_metrics, s_img, f_img, f_hist, t_hist)
            else:
                print(
                    f"ERROR: Skipping source {s} as its size does not match with fake {f}")

        df = pd.concat(
            {m: pd.DataFrame.from_dict(data=metrics, orient='index')},
            names=['Model'], axis=1)
        metrics_df = pd.concat([metrics_df, df], axis=1)

    if not metrics_df.empty:
        metrics_df.index.name = "Source"
        metrics_df.sort_index(inplace=True)
        ut.multicolumn2csv(metrics_df,
                           str(results_path / f"{ut.path2fn(str(metrics_conf.source))}.csv"))

    return metrics_df


def unpaired_WD(path1: str, path2: str):
    """
    Computes WD for unpaired ditributions.
    The average histogram for each distribution is first calculated and then WD
    from these two histograms is computed.

    :param path1: Directory containing first distribution images.
    :param path2: Directory containing second distribution images.
    """

    path1, path2 = Path(path1), Path(path2)
    av_hists = []
    for p in [path1, path2]:
        ut.check_path(p)
        files = get_img_files(p)
        hist = np.zeros((3, 256))
        for f in files:
            img = cv2.imread(f, 1)
            hist += get_hist(img)
        av_hists.append(hist / len(files))

    return av_wasserstein(av_hists[0], av_hists[1])


def collect_distributions(path: str, chr1: np.array, chr2: np.array, bins: int):
    img_files = get_img_files(path)
    for img_file in img_files:
        img = Image.open(img_file)
        lab = color.rgb2lab(img)
        chr1_values = np.clip(np.ravel(lab[:, :, 1]), -128, 127)
        chr2_values = np.clip(np.ravel(lab[:, :, 2]), -128, 127)
        chr1 += np.histogram(chr1_values, bins=bins)[0]
        chr2 += np.histogram(chr2_values, bins=bins)[0]

    return chr1, chr2


def normalize_pop(h1: np.array, step=1):
    return h1 / np.sum(h1) / step


def unpaired_lab_WD(p1: str, p2: str):
    step = 1
    bins = np.arange(-128, 128, step)
    chr1_pop1 = np.zeros(len(bins) - 1)
    chr2_pop1, chr1_pop2, chr2_pop2 = chr1_pop1.copy(), chr1_pop1.copy(), chr1_pop1.copy()

    ut.check_path(p1)
    ut.check_path(p2)
    chr1_pop1, chr2_pop1 = collect_distributions(p1, chr1_pop1, chr2_pop1, bins)
    chr1_pop2, chr2_pop2 = collect_distributions(p2, chr1_pop2, chr2_pop2, bins)

    if chr1_pop1.sum() == 0 or chr2_pop1.sum() == 0:
        print('first population is not yet gathered')
        return None
    if chr1_pop2.sum() == 0 or chr2_pop2.sum() == 0:
        print('second population is not yet gathered')
        return None

    distance_chr1 = wasserstein_distance(normalize_pop(chr1_pop1),
                                         normalize_pop(chr1_pop2))
    distance_chr2 = wasserstein_distance(normalize_pop(chr2_pop1),
                                         normalize_pop(chr2_pop2))

    return max(distance_chr1, distance_chr2)


def unpaired_lab_WD(p1: str, p2: str):
    """
    Computes LAB WD for unpaired distributions.

    :param path1: Directory containing first distribution images.
    :param path2: Directory containing second distribution images.
    """

    step = 1
    bins = np.arange(-128, 128, step)
    chr1 = np.zeros((2, len(bins) - 1))
    chr2 = chr1.copy()
    av_hists = []
    for i, path in enumerate([p1, p2]):
        ut.check_path(path)
        img_files = get_img_files(path)
        hist = np.zeros((3, 256))
        for img_file in img_files:
            hist += get_hist(cv2.imread(img_file, 1))
            lab = color.rgb2lab(Image.open(img_file))
            chr1_values = np.clip(np.ravel(lab[:, :, 1]), -128, 127)
            chr2_values = np.clip(np.ravel(lab[:, :, 2]), -128, 127)
            chr1[i] += np.histogram(chr1_values, bins=bins)[0]
            chr2[i] += np.histogram(chr2_values, bins=bins)[0]
        av_hists.append(hist / len(img_files))

        if chr1[i].sum() == 0 or chr2[i].sum() == 0:
            print(f'population {i + 1} is not yet gathered')
            return None

    lab_wd = max(
        wasserstein_distance(normalize_pop(chr1[0]), normalize_pop(chr1[1])),
        wasserstein_distance(normalize_pop(chr2[0]), normalize_pop(chr2[1])))

    return lab_wd


def calculate_fid(paths: list, batch_size: int, device: str, dims: int = 512,
                  num_workers: int = 4):
    """Calculates the FID of two paths.

    :param path: Path containing images.
    :param batch_size: Batch size of images for the model to process at once.
    :param device: Device to run calculations.
    :param dims: z dimensionality.
    :param num_workers: Number workers.
    :return: FID distance.
    """

    return fid.compute_fid(paths[0], paths[1], mode="clean",
                           num_workers=num_workers,
                           batch_size=batch_size, device=device, z_dim=dims)
