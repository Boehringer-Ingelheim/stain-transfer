import numpy as np
import torch
import torchvision.transforms as transforms

from src.modelling.base_model import BaseModel
from src.modelling.networks import ResnetGenerator, NORM_LAYER


class CycleGan(BaseModel):
    """
    This class implements the CycleGAN model generator, for image-to-image
    translation inference. CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf.
    """

    no_antialias = True
    no_antialias_up = True

    def __init__(self, conf):
        """
        Initializes model and create dataset loaders.

        :param conf: See BaseModel.
        :param no_antialias: leave as True, do not modify.
        :param no_antialias_up: leave as True, do not modify.
        """

        BaseModel.__init__(self, conf)
        self.net = ResnetGenerator(3, 3, 64, norm_layer=NORM_LAYER,
                                   use_dropout=False, n_blocks=9,
                                   no_antialias=self.__class__.no_antialias,
                                   no_antialias_up=self.__class__.no_antialias_up)
        self.net.to(self.conf.device)
        self.load_weights()
        self.dataloader = self.create_dataloader()

    def transform(self):
        """ Pre-process transformations to be applied to inputs in data loaders."""
        return [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def forward(self, t: torch.Tensor):
        """ Run forward pass."""
        return self.net(t.to(self.conf.device))

    def post_transform(self, t: torch.Tensor):
        """ Post-process transformations to be applied to network outputs."""
        t = np.transpose(t.detach().cpu().float().numpy(), (0, 2, 3, 1))

        return np.clip((t + 1) / 2.0 * 255, 0, 255).astype(np.uint8)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """ Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (
                    key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (
                    key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1)

    def load_weights(self):
        state_dict = torch.load(self.conf.weights,
                                map_location=self.conf.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):
            # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.net,
                                                  key.split('.'))
        self.net.load_state_dict(state_dict)
