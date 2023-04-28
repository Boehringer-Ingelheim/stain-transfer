from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from src.data_processing.datasets import ImagePathDataset
from src.modelling.base_model import BaseModel


class RGB2LAB(object):
    """ Converts an RGB PIL.Image to a torch.FloatTensor"""

    def __call__(self, pic: Image):
        """
        :param pic: Image to be converted to tensor.
        :return LAB: LAB tensor.
        """

        lab = cv2.cvtColor(np.array(pic), cv2.COLOR_RGB2LAB).astype(np.float32)

        return torch.from_numpy(lab)


class RunningStats():
    """Implements Welford's algorithm for running averages """

    def __init__(self):
        """
        Initialize running stats.

        count: aggregates the number of samples seen so far
        mean: accumulates the mean of the entire dataset
        M2: aggregates the squared distance from the mean
        """

        self.count = 0
        self.mean = 0
        self.M2 = 0

    def update(self, newValue):
        """For a new value newValue, update count, mean and M2."""

        self.count += 1
        delta = newValue - self.mean
        self.mean += delta / self.count
        delta2 = newValue - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        """Retrieve the mean, variance and sample variance."""
        if self.count < 2:
            return self.mean, self.M2 / self.count, float("nan")
        else:
            return self.mean, self.M2 / self.count, self.M2 / (self.count - 1)


class ColorStat(BaseModel):
    MODE = {1: "computing means and stds from configuration target_path",
            2: "using means and std from configuration target_tensor",
            3: "using means and std from configuration weights"}

    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.dataloader = self.create_dataloader()
        self.__post_init__()

    def __post_init__(self):
        if self.conf.target_path is not None:
            if self.conf.target_samples is None:
                msg = f"Target samples not specified for {self.__class__.__name__} method."
                raise AssertionError(msg)
            self.create_target_iterator()
            mode = 1
            if self.conf.target_samples == -1:
                self.full_target_train()
        elif self.conf.target_tensor is not None:
            self.load_average(self.conf.target_tensor)
            mode = 2
        else:
            self.load_average(self.conf.weights)
            mode = 3
        if 'averagetensor' not in self.__class__.__name__.lower():
            print(f'{self.__class__.__name__} inference mode type:'
                  f' {self.__class__.MODE[mode]}.')

    def full_target_train(self):
        self.t_mean, self.t_std = self.get_average()

    def create_target_iterator(self):
        """ Creates target path iterator."""
        transforms_list = self.transform()
        if self.conf.center_crop:
            center_crop = transforms.CenterCrop(self.conf.center_crop)
            transforms_list.insert(0, center_crop)
        target_dataset = ImagePathDataset(self.conf.target_path,
                                          transforms.Compose(transforms_list))
        ts = 1 if self.conf.target_samples == -1 else self.conf.target_samples
        nw = 0 if self.conf.target_samples == -1 else self.conf.num_workers

        target_dataloader = torch.utils.data.DataLoader(target_dataset,
                                                        batch_size=ts,
                                                        shuffle=True,
                                                        drop_last=False,
                                                        num_workers=nw)
        self.target_iterator = iter(target_dataloader)

    def get_targets(self):
        """Gets a batch of target images. Restart iterator if no more targets
        are left."""

        try:
            targets, filenames = next(self.target_iterator)
        except StopIteration:
            self.create_target_iterator()
            targets, filenames = next(self.target_iterator)

        return targets.to(self.conf.device), filenames

    def get_average(self):
        running_means = [RunningStats(), RunningStats(), RunningStats()]
        running_stds = [RunningStats(), RunningStats(), RunningStats()]
        print(f'Computing average means and stds for {self.conf.target_path}')
        progress_bar = tqdm(total=len(self.target_iterator))
        while True:
            try:
                target = next(self.target_iterator)[0]
                for t in target:
                    progress_bar.update(1)
                    means = t.mean(axis=(0, 1))
                    stds = t.std(axis=(0, 1))
                    for i in range(3):
                        running_means[i].update(means[i])
                        running_stds[i].update(stds[i])
            except StopIteration:
                del self.target_iterator
                break
        progress_bar.close()
        means = [running_means[i].get_stats()[0] for i in range(3)]
        stds = [running_stds[i].get_stats()[0] for i in range(3)]

        return torch.tensor([[[means]]]), torch.tensor([[[stds]]])

    def load_average(self, average_tensor_path: str):
        s = torch.load(average_tensor_path)
        self.t_mean, self.t_std = s['means'], s['stds']

    def transform(self):
        """Pre-process transformations to be applied to inputs in data loaders."""
        return [RGB2LAB()]

    def forward(self, x: torch.Tensor):
        """ Run forward pass."""

        if self.conf.target_samples != -1 and self.conf.target_path is not None:
            t = self.get_targets()[0]
            self.t_mean = t.mean(axis=(1, 2)).mean(axis=0).view(1, 1, 1, 3)
            self.t_std = t.std(axis=(1, 2)).mean(axis=0).view(1, 1, 1, 3)

        samples = []
        for xt in x:
            mean = xt.mean(axis=(0, 1))
            std = xt.std(axis=(0, 1))
            samples.append((xt - mean) * (self.t_std / std) + self.t_mean)
        samples = torch.clip(torch.cat(samples).round(), min=0, max=255)

        return samples

    def post_transform(self, x: torch.Tensor):
        """ Post-process transformations applied to outputs."""
        x = x.detach().cpu().numpy().astype(np.uint8)

        return np.array([cv2.cvtColor(img, cv2.COLOR_LAB2RGB) for img in x])


class ColorStatAverageTensor(ColorStat):
    """ColorStatAverageTensor"""

    def __init__(self, conf):
        conf.target_samples = -1
        conf.target_path = conf.data_path
        ColorStat.__init__(self, conf)
        dp = Path(self.conf.data_path)
        filename = '_'.join(dp.parts[1:] if dp.is_absolute() else dp.parts)
        self.filename = (self.image_outs / filename).with_suffix('.pth')

    def predict(self):
        self.image_outs.mkdir(parents=True, exist_ok=True)
        torch.save({"means": self.t_mean, "stds": self.t_std}, self.filename)
        print(f"Saved means and stds to {self.filename}")
        print(f"Computed:\n  means: {self.t_mean} \n  stds: {self.t_std}")
