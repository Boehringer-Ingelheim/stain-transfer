from pathlib import Path

import torch
from tqdm import tqdm

from src.modelling.networks import MunitGenerator
from src.modelling.unit import Unit

MODE = {1: "computing styles from configuration target_path",
        2: "using style from configuration target_tensor",
        3: "using random styles"}


class Munit(Unit):
    def __init__(self, conf):
        Unit.__init__(self, conf, gen=MunitGenerator)
        if conf.target_path:
            self.start_targer_iter()
            self.target_style = None
            mode = 1
        elif conf.target_tensor:
            self.target_iterator = None
            self.target_style = torch.load(conf.target_tensor,
                                            map_location=self.conf.device)
            mode = 2
        else:
            self.target_iterator = None
            self.target_style = None
            mode = 3
        if 'averagetensor' not in self.__class__.__name__.lower():
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

    def forward(self, t: torch.Tensor):
        target_samples = None
        if self.target_iterator:
            target_samples = self.get_targets()
            target_samples = target_samples[:t.shape[0]]
        return self.net(t.to(self.conf.device), a2b=self.conf.a2b,
                        style_images=target_samples,
                        target_style=self.target_style,
                        device=self.conf.device)


class MunitAverageTensor(Munit):
    """MunitAverageTensor"""

    def __init__(self, conf):
        Unit.__init__(self, conf, gen=MunitGenerator)
        if self.conf.a2b:
            self.style_encode = self.net.autoencoder_a
        else:
            self.style_encode = self.net.autoencoder_b
        dp = Path(self.conf.data_path)
        filename = '_'.join(dp.parts[1:] if dp.is_absolute() else dp.parts)
        self.filename = (self.image_outs / filename).with_suffix('.pth')

    def predict(self):
        self.image_outs.mkdir(parents=True, exist_ok=True)
        self.eval()
        style = torch.zeros(self.style_encode.style_channels, 1, 1,
                            device=torch.device(self.conf.device))
        print(f'Computing average style tensor for {self.conf.data_path}')
        total = 0
        with torch.no_grad():
            for imgs, imgs_path in tqdm(self.dataloader):
                imgs = imgs.to(self.conf.device)
                batch_style = self.style_encode.style_encoder(imgs)
                style += batch_style.sum(axis=0)
                total += len(imgs)
        style /= total
        torch.save(style.unsqueeze(0), self.filename)
        print(f"Total styles computed: {total}")
        print(f"Average style {style}")
        print(f"Saved average style to {self.filename}")
