import torch
from typing import Union
from src.modelling.base_model import BaseModel
from src.modelling.cyclegan import CycleGan
from src.modelling.networks import UnitGenerator, MunitGenerator


class Unit(CycleGan):
    def __init__(self, conf,
                 gen: Union[UnitGenerator, MunitGenerator] = UnitGenerator):
        BaseModel.__init__(self, conf)
        self.keep = 'module.module.'
        self.net = gen()
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()

    def forward(self, t: torch.Tensor):
        return self.net(t.to(self.conf.device), a2b=self.conf.a2b)

    def load_weights(self):
        state_dict = torch.load(self.conf.weights,
                                map_location=self.conf.device)
        state_dict = {k.replace(self.keep, '', 1): v for k, v in
                      state_dict.items() if k.startswith(self.keep)}
        self.net.load_state_dict(state_dict)
