from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from src.modelling.base_model import BaseModel
from src.modelling.networks import DritNet

MODE = {1: "computing attributes from configuration target_path",
        2: "using attributes from configuration target_tensor",
        3: "using random attributes"}


class Drit(BaseModel):
    """This class implements the DRIT model encoders and generators for
    image-to-image translation inference.
    DRIT paper: https://arxiv.org/pdf/1808.00948.pdf.
    """

    def __init__(self, conf: dict):
        """ Initializes model and create dataset loaders."""
        BaseModel.__init__(self, conf)
        self.net = DritNet()
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()
        if 'averagetensor' not in self.__class__.__name__.lower():
            if conf.target_path:
                self.start_targer_iter()
                self.target_attr = None
                mode = 1
            elif conf.target_tensor:
                self.target_iterator = None
                self.target_attr = torch.load(conf.target_tensor,
                                              map_location=self.conf.device)
                mode = 2
            else:
                self.target_iterator = None
                self.target_attr = None
                mode = 3
            print(f'{self.__class__.__name__} inference mode type: '
                  f'{MODE[mode]}.')

    def start_targer_iter(self):
        self.target_iterator = iter(self.create_dataloader(
            data_path=self.conf.target_path, shuffle=True, drop_last=True))

    def get_targets(self):
        """Gets a batch of target images. Restart iterator if no more targets
        are left."""

        try:
            targets = next(self.target_iterator)[0]
        except StopIteration:
            self.start_targer_iter()
            targets = next(self.target_iterator)[0]

        return targets.to(self.conf.device)

    def transform(self):
        """ Pre-process transformations to be applied to inputs in data loaders."""
        return [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def forward(self, t: torch.Tensor):
        """ Run forward pass."""
        target_samples, target_attr = None, None
        if self.target_iterator:
            target_samples = self.get_targets()
            target_samples = target_samples[:t.shape[0]]
        if self.target_attr is not None:
            target_attr = self.target_attr.repeat(len(t), 1)
        return self.net(t.to(self.conf.device), a2b=self.conf.a2b,
                        attr_images=target_samples,
                        target_attr=target_attr,
                        device=self.conf.device)

    def post_transform(self, t: torch.Tensor):
        """ Post-process transformations to be applied to network outputs."""
        t = np.transpose(t.detach().cpu().float().numpy(), (0, 2, 3, 1))

        return np.clip((t + 1) / 2.0 * 255, 0, 255).astype(np.uint8)

    def load_weights(self):
        self.net.load_weights(self.conf.weights, self.conf.device)


class DritAverageTensor(Drit):
    """DritAverageTensor"""

    def __init__(self, conf):
        Drit.__init__(self, conf)
        if self.conf.a2b:
            self.attr_enc = self.net.enc_a.forward_a
        else:
            self.attr_enc = self.net.enc_a.forward_b
        dp = Path(self.conf.data_path)
        filename = '_'.join(dp.parts[1:] if dp.is_absolute() else dp.parts)
        self.filename = (self.image_outs / filename).with_suffix('.pth')

    def predict(self):
        self.image_outs.mkdir(parents=True, exist_ok=True)
        self.eval()
        attr = torch.zeros(1, self.net.nz).to(self.conf.device)
        print(f'Computing average attribute tensor for {self.conf.data_path}')
        total = 0
        with torch.no_grad():
            for imgs, imgs_path in tqdm(self.dataloader):
                imgs = imgs.to(self.conf.device)
                batch_attr = self.net.get_img_attr(self.attr_enc, imgs)
                attr += batch_attr.sum(axis=0)
                total += len(imgs)
        attr /= total
        torch.save(attr, self.filename)
        print(f"Total attrs computed: {total}")
        print(f"Average attr {attr}")
        print(f"Saved average attr to {self.filename}")
