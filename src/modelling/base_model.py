from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from src.data_processing.datasets import ImagePathDataset
from src.modelling import (RESULTS_DIR, IMAGE_OUT_DIR, NAMELESS_MODEL,
                           AV_TENSOR_OUT_DIR)

if TYPE_CHECKING:
    from src.data_processing.config import GenerateCOnf


class BaseModel(ABC):
    """
    This class is an abstract base class (ABC) for models. To create a subclass,
    you need to implement the following methods:
    __init__: initialize the class; first call BaseModel.__init__(self, conf).
    transform: apply specific model pre-processing.
    forward: model inference.
    post_transform: apply specific model post-processing.
    """

    def __init__(self, conf: GenerateCOnf):
        """
        Initialize the BaseModel class. When creating your custom class, you
        need to implement your own initialization, which must move model to device
        and create dataloader. In this function, you should first call
        <BaseModel.__init__(self, conf)>.

        :param conf: inference configuration object.
        """

        self.net = None
        self.conf = conf
        results_path = Path(conf.results_path).absolute()
        if results_path != RESULTS_DIR:
            self.image_outs = results_path / self.__class__.__name__
        else:
            dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
            out = AV_TENSOR_OUT_DIR if 'averagetensor' in self.__class__.__name__.lower() else IMAGE_OUT_DIR
            self.image_outs = out / self.__class__.__name__ / dt
        if conf.add_suffix:
            self.name = f"_{self.__class__.__name__}"
        else:
            self.name = NAMELESS_MODEL

    @abstractmethod
    def transform(self):
        """
        Pre-process transformations to be applied to inputs.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward.
        """
        pass

    @abstractmethod
    def post_transform(self):
        """
        Post-process transformations to be applied to outputs.
        """
        pass

    def save_outputs(self, np_images: np.ndarray, paths: tuple):
        """ Saves image results."""
        for img, img_path in zip(np_images, paths):
            fn = f'{Path(img_path).stem}{self.name}{Path(img_path).suffix}'
            Image.fromarray(img).save(self.image_outs / fn)

    def eval(self):
        if self.net: self.net.eval()

    def create_dataloader(self, data_path: str = None, shuffle: bool = False,
                          drop_last: bool = False):
        """ Creates dataloader."""
        if not data_path: data_path = self.conf.data_path
        transforms_list = self.transform()
        if self.conf.center_crop:
            center_crop = transforms.CenterCrop(self.conf.center_crop)
            transforms_list.insert(0, center_crop)
        dataset = ImagePathDataset(data_path,
                                   transforms.Compose(transforms_list))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.conf.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.conf.num_workers)

        return dataloader

    def predict(self):
        """
        Forward function used in test time. This function sets the model to
        eval mode and wraps <forward> function in no_grad() so we don't save
        intermediate steps for backprop. It also calls <save_outputs> to store results.
        """
        self.image_outs.mkdir(parents=True, exist_ok=True)
        self.eval()
        with torch.no_grad():
            for imgs, imgs_path in tqdm(self.dataloader):
                image_numpy = self.post_transform(self.forward(imgs))
                self.save_outputs(image_numpy, imgs_path)
