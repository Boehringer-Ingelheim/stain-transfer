import functools
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

NORM_LAYER = functools.partial(nn.InstanceNorm2d, affine=False,
                               track_running_stats=False)
LRELU = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_filter(filt_size=3):
    if (filt_size == 1):
        a = np.array([1., ])
    elif (filt_size == 2):
        a = np.array([1., 1.])
    elif (filt_size == 3):
        a = np.array([1., 2., 1.])
    elif (filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif (filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif (filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif (filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class UnetSCBlock(nn.Module):
    """Unet Skip Connection Block: Defines the Unet submodule with skip connection."""

    def __init__(self, outer_nc: int, inner_nc: int, input_nc: int = None,
                 submodule=None,
                 outermost: bool = False, innermost: bool = False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout: bool = False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            :param outer_nc: number of filters in the outer conv layer
            :param inner_nc: number of filters in the inner conv layer
            :param input_nc: number of channels in input images/features
            :param submodule: previously defined UnetSCBlock submodules
            :param outermost: if this module is the outermost module
            :param innermost: if this module is the innermost module
            :param norm_layer: normalization layer
            :param use_dropout: if use dropout layers.
        """

        super(UnetSCBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
                             padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                                        stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4,
                                        stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc: int, output_nc: int, num_downs: int,
                 ngf: int = 64, norm_layer=nn.BatchNorm2d,
                 use_dropout: bool = False):
        """Construct a Unet generator

        :param input_nc:   number of channels in input images
        :param output_nc:  number of channels in output images
        :param num_downs:  number of downsamplings in UNet. For example,
        if |num_downs| == 7, image of size 128x128 will become of size 1x1
        at the bottleneck
        :param ngf:        number of filters in the last conv layer
        :param norm_layer: normalization layer

        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSCBlock(ngf * 8, ngf * 8, input_nc=None,
                                 submodule=None, norm_layer=norm_layer,
                                 innermost=True)
        for i in range(num_downs - 5):
            # add intermediate layers with ngf * 8 filters
            unet_block = UnetSCBlock(ngf * 8, ngf * 8, input_nc=None,
                                     submodule=unet_block,
                                     norm_layer=norm_layer,
                                     use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSCBlock(ngf * 4, ngf * 8, input_nc=None,
                                 submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSCBlock(ngf * 2, ngf * 4, input_nc=None,
                                 submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSCBlock(ngf, ngf * 2, input_nc=None,
                                 submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSCBlock(output_nc, ngf, input_nc=input_nc,
                                 submodule=unet_block, outermost=True,
                                 norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Initialize the Resnet block
        A resnet block is a conv block with skip connections. We construct a
        conv block with build_conv_block function, and implement skip connections
        in <forward> function. Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self, dim: int, padding_type: str,
                         norm_layer: functools.partial, use_dropout: bool,
                         use_bias: bool):
        """
        Construct a convolutional block.

        :param dim: number of channels in the conv layer.
        :param padding_type: name of padding layer: reflect | replicate | zero
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers.
        :param use_bias:    if the conv layer uses bias or not
        :return: a conv block (with a conv layer, a normalization layer, and a
        non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """ Resnet-based generator that consists of Resnet blocks between a few
    downsampling/upsampling operations."""

    def __init__(self, input_nc: int, output_nc: int, ngf: int = 64,
                 norm_layer: functools.partial = nn.BatchNorm2d,
                 use_dropout: bool = False, n_blocks: int = 6,
                 padding_type: str = 'reflect', no_antialias: bool = True,
                 no_antialias_up: bool = True):
        """Construct a Resnet-based generator

        :param input_nc: number of channels in input images
        :param output_nc: number of channels in output images
        :param ngf: number of filters in the last conv layer
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers
        :param n_blocks: the number of ResNet blocks
        :param padding_type: the name of padding layer in conv layers:
        reflect | replicate | zero
        :param no_antialias: if true, use stride=2 convs instead of
        antialiased-downsampling
        :param no_antialias_up: if true, use [upconv(learned filter)] instead of
        [upconv(hard-coded [1,3,3,1] filter), conv]
        """

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2), nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2), nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2, padding=1,
                                             output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        """Standard forward"""
        return self.model(input)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer('filt', filt[None, None, :, :].repeat(
            (self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = nn.functional.conv_transpose2d(
            self.pad(inp), self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1])[:, :, 1:, 1:]
        if (self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2,
                 pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat(
            (self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return nn.functional.conv2d(self.pad(inp), self.filt,
                                        stride=self.stride, groups=inp.shape[1])


class StainNetNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32,
                 kernel_size=1):
        super(StainNetNet, self).__init__()
        model_list = []
        model_list.append(nn.Conv2d(input_nc,
                                    n_channel,
                                    kernel_size=kernel_size,
                                    bias=True,
                                    padding=kernel_size // 2))
        model_list.append(nn.ReLU(True))
        for n in range(n_layer - 2):
            model_list.append(nn.Conv2d(n_channel,
                                        n_channel,
                                        kernel_size=kernel_size,
                                        bias=True,
                                        padding=kernel_size // 2))
            model_list.append(nn.ReLU(True))
        model_list.append(nn.Conv2d(n_channel,
                                    output_nc,
                                    kernel_size=kernel_size,
                                    bias=True,
                                    padding=kernel_size // 2))

        self.rgb_trans = nn.Sequential(*model_list)

    def forward(self, x):
        return self.rgb_trans(x)


class Drit_E_content(nn.Module):
    def __init__(self, input_dim_a, input_dim_b):
        super(Drit_E_content, self).__init__()
        encA_c = []
        tch = 64
        encA_c += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1,
                                   padding=3)]
        for i in range(1, 3):
            encA_c += [
                ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0, 3):
            encA_c += [INSResBlock(tch, tch)]

        encB_c = []
        tch = 64
        encB_c += [LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1,
                                   padding=3)]
        for i in range(1, 3):
            encB_c += [
                ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0, 3):
            encB_c += [INSResBlock(tch, tch)]

        enc_share = []
        for i in range(0, 1):
            enc_share += [INSResBlock(tch, tch)]
            enc_share += [GaussianNoiseLayer()]
            self.conv_share = nn.Sequential(*enc_share)

        self.convA = nn.Sequential(*encA_c)
        self.convB = nn.Sequential(*encB_c)

    def forward_a(self, xa):
        outputA = self.convA(xa)
        outputA = self.conv_share(outputA)
        return outputA

    def forward_b(self, xb):
        outputB = self.convB(xb)
        outputB = self.conv_share(outputB)
        return outputB


class Drit_E_attr_concat(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_nc=8, norm_layer=None,
                 nl_layer=None):
        super(Drit_E_attr_concat, self).__init__()

        ndf = 64
        n_blocks = 4
        max_ndf = 4

        conv_layers_A = [nn.ReflectionPad2d(1)]
        conv_layers_A += [
            nn.Conv2d(input_dim_a, ndf, kernel_size=4, stride=2, padding=0,
                      bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_A += [
                BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_A = nn.Sequential(*conv_layers_A)

        conv_layers_B = [nn.ReflectionPad2d(1)]
        conv_layers_B += [
            nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0,
                      bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_B += [
                BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_B = nn.Sequential(*conv_layers_B)

    def forward_a(self, xa):
        x_conv_A = self.conv_A(xa)
        conv_flat_A = x_conv_A.view(xa.size(0), -1)
        output_A = self.fc_A(conv_flat_A)
        outputVar_A = self.fcVar_A(conv_flat_A)
        return output_A, outputVar_A

    def forward_b(self, xb):
        x_conv_B = self.conv_B(xb)
        conv_flat_B = x_conv_B.view(xb.size(0), -1)
        output_B = self.fc_B(conv_flat_B)
        outputVar_B = self.fcVar_B(conv_flat_B)
        return output_B, outputVar_B


class Drit_G_concat(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, nz):
        super(Drit_G_concat, self).__init__()
        self.nz = nz
        tch = 256
        dec_share = []
        dec_share += [INSResBlock(tch, tch)]
        self.dec_share = nn.Sequential(*dec_share)
        tch = 256 + self.nz
        decA1 = []
        for i in range(0, 3):
            decA1 += [INSResBlock(tch, tch)]
        tch = tch + self.nz
        decA2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decA3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decA4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1,
                                    padding=0)] + [nn.Tanh()]
        self.decA1 = nn.Sequential(*decA1)
        self.decA2 = nn.Sequential(*[decA2])
        self.decA3 = nn.Sequential(*[decA3])
        self.decA4 = nn.Sequential(*decA4)

        tch = 256 + self.nz
        decB1 = []
        for i in range(0, 3):
            decB1 += [INSResBlock(tch, tch)]
        tch = tch + self.nz
        decB2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decB3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decB4 = [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1,
                                    padding=0)] + [nn.Tanh()]
        self.decB1 = nn.Sequential(*decB1)
        self.decB2 = nn.Sequential(*[decB2])
        self.decB3 = nn.Sequential(*[decB3])
        self.decB4 = nn.Sequential(*decB4)

    def forward_a(self, x, z):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                          x.size(2), x.size(3))
        x_and_z = torch.cat([out0, z_img], 1)
        out1 = self.decA1(x_and_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                           out1.size(2),
                                                           out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decA2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                           out2.size(2),
                                                           out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decA3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                           out3.size(2),
                                                           out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decA4(x_and_z4)
        return out4

    def forward_b(self, x, z):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                          x.size(2), x.size(3))
        x_and_z = torch.cat([out0, z_img], 1)
        out1 = self.decB1(x_and_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                           out1.size(2),
                                                           out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decB2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                           out2.size(2),
                                                           out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decB3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1),
                                                           out3.size(2),
                                                           out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decB4(x_and_z4)
        return out4


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0,
                  bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0,
                      bias=True)]


class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return nn.functional.layer_norm(x, normalized_shape,
                                            self.weight.expand(
                                                normalized_shape),
                                            self.bias.expand(normalized_shape))
        else:
            return nn.functional.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None',
                 sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride,
                            padding=0, bias=True)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride,
                            padding=0, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1),
                nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self, ):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise


class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding,
                 output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size,
                                     stride=stride, padding=padding,
                                     output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DritNet(nn.Module):
    def __init__(self):
        super(DritNet, self).__init__()
        self.nz = 8
        self.enc_c = Drit_E_content(3, 3)
        self.enc_a = Drit_E_attr_concat(3, 3, self.nz, norm_layer=None,
                                        nl_layer=LRELU)
        self.gen = Drit_G_concat(3, 3, nz=self.nz)

    def forward(self, content_images, a2b=True, attr_images=None,
                target_attr=None, device=torch.device('cpu')):

        if a2b:
            content_encode = self.enc_c.forward_a
            attr_enc = self.enc_a.forward_b
            gen = self.gen.forward_b
        else:
            content_encode = self.enc_c.forward_b
            attr_enc = self.enc_a.forward_a
            gen = self.gen.forward_a

        z_content = content_encode(content_images)
        if attr_images is not None:
            attr = self.get_img_attr(attr_enc, attr_images)
        elif target_attr is not None:
            attr = target_attr
        else:
            attr = torch.randn(content_images.size(0), self.nz).to(device)

        output = gen(z_content, attr)

        return output

    def get_img_attr(self, attr_encoder, attr_images):
        mu, logvar = attr_encoder(attr_images)
        return logvar.mul(0.5).exp() + mu

    def load_weights(self, weights: str, device):
        state_dict = torch.load(weights, map_location=device)
        self.enc_c.load_state_dict(state_dict['enc_c'])
        self.enc_a.load_state_dict(state_dict['enc_a'])
        self.gen.load_state_dict(state_dict['gen'])


class AdaptiveNorm(nn.Module):
    def __init__(self, num_features, cond_dims, weight_norm_type='',
                 projection=True, projection_bias=True,
                 separate_projection=False, input_dim=2,
                 activation_norm_type='instance', activation_norm_params=None,
                 apply_noise=False, add_bias=True, input_scale=1.0,
                 init_gain=1.0):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type, input_dim,
                                              **vars(activation_norm_params))

        if not apply_noise: self.noise_layer = None
        if projection:
            if separate_projection:
                self.fc_gamma = LinearBlock(cond_dims, num_features,
                                            weight_norm_type=weight_norm_type,
                                            bias=projection_bias)
                self.fc_beta = LinearBlock(cond_dims, num_features,
                                           weight_norm_type=weight_norm_type,
                                           bias=projection_bias)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2,
                                      weight_norm_type=weight_norm_type,
                                      bias=projection_bias)

        self.projection = projection
        self.separate_projection = separate_projection
        self.input_scale = input_scale
        self.add_bias = add_bias
        self.conditional = True
        self.init_gain = init_gain

    def forward(self, x, y, noise=None, **_kwargs):
        y = y * self.input_scale
        if self.projection:
            if self.separate_projection:
                gamma = self.fc_gamma(y)
                beta = self.fc_beta(y)
                for _ in range(x.dim() - gamma.dim()):
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
            else:
                y = self.fc(y)
                for _ in range(x.dim() - y.dim()):
                    y = y.unsqueeze(-1)
                gamma, beta = y.chunk(2, 1)
        else:
            for _ in range(x.dim() - y.dim()):
                y = y.unsqueeze(-1)
            gamma, beta = y.chunk(2, 1)
        if self.norm is not None:
            x = self.norm(x)
        if self.noise_layer is not None:
            x = self.noise_layer(x, noise=noise)
        if self.add_bias:
            x = torch.addcmul(beta, x, 1 + gamma)
            return x
        else:
            return x * (1 + gamma), beta.squeeze(3).squeeze(2)


def get_activation_norm_layer(num_features, norm_type, input_dim,
                              **norm_params):
    input_dim = max(input_dim, 1)  # Norm1d works with both 0d and 1d inputs
    if norm_type == 'none' or norm_type == '':
        norm_layer = None
    elif norm_type == 'instance':
        affine = norm_params.pop('affine', True)  # Use affine=True by default
        norm = getattr(nn, 'InstanceNorm%dd' % input_dim)
        norm_layer = norm(num_features, affine=affine, **norm_params)
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
    else:
        raise ValueError(f'Activation norm layer {norm_type} unavailable')

    return norm_layer


def get_weight_norm_layer(norm_type, **norm_params):
    if norm_type == 'none' or norm_type == '':  # no normalization
        return lambda x: x
    elif norm_type == 'spectral':  # spectral normalization
        return functools.partial(nn.utils.spectral_norm, **norm_params)
    else:
        raise ValueError(f'Weight norm layer {norm_type} unavailable')


def get_nonlinearity_layer(nonlinearity_type, inplace, **kwargs):
    if nonlinearity_type == 'relu':
        nonlinearity = nn.ReLU(inplace=inplace)
    elif nonlinearity_type == 'tanh':
        nonlinearity = nn.Tanh()
    elif nonlinearity_type == 'none' or nonlinearity_type == '':
        nonlinearity = None
    else:
        raise ValueError(f'Nonlinearity {nonlinearity_type} unavailable')

    return nonlinearity


class _BaseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params, activation_norm_type,
                 activation_norm_params,
                 skip_activation_norm, skip_nonlinearity, nonlinearity,
                 inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels, order, block,
                 learn_shortcut, clamp, output_scale, skip_block=None,
                 blur=False, upsample_first=True, skip_weight_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale = output_scale
        self.upsample_first = upsample_first
        self.stride = stride
        self.blur = blur
        if skip_block is None:
            skip_block = block

        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            # The bias for conv_block_0, conv_block_1, and conv_block_s.
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError('Bias list must be 3.')
        else:
            raise ValueError('Bias must be either an integer or s list.')
        if learn_shortcut is None:
            self.learn_shortcut = (in_channels != out_channels)
        else:
            self.learn_shortcut = learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)

        # Parameters.
        residual_params = {}
        shortcut_params = {}
        base_params = dict(dilation=dilation, groups=groups,
                           padding_mode=padding_mode, clamp=clamp)
        residual_params.update(base_params)
        residual_params.update(
            dict(activation_norm_type=activation_norm_type,
                 activation_norm_params=activation_norm_params,
                 weight_norm_type=weight_norm_type,
                 weight_norm_params=weight_norm_params, padding=padding,
                 apply_noise=apply_noise))
        shortcut_params.update(base_params)
        shortcut_params.update(dict(kernel_size=1))
        if skip_activation_norm:
            shortcut_params.update(
                dict(activation_norm_type=activation_norm_type,
                     activation_norm_params=activation_norm_params,
                     apply_noise=False))
        if skip_weight_norm:
            shortcut_params.update(dict(weight_norm_type=weight_norm_type,
                                        weight_norm_params=weight_norm_params))

        # Residual branch.
        if order.find('A') < order.find('C') and \
                (activation_norm_type == '' or activation_norm_type == 'none'):
            # Nonlinearity is the first operation in the residual path.
            # In-place nonlinearity will modify the input variable and cause
            # backward error.
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity

        (first_stride, second_stride, shortcut_stride,
         first_blur, second_blur, shortcut_blur) = self._get_stride_blur()
        self.conv_block_0 = block(in_channels, hidden_channels,
                                  kernel_size=kernel_size, bias=biases[0],
                                  nonlinearity=nonlinearity, order=order[0:3],
                                  inplace_nonlinearity=first_inplace,
                                  stride=first_stride, blur=first_blur,
                                  **residual_params)
        self.conv_block_1 = block(hidden_channels, out_channels,
                                  kernel_size=kernel_size, bias=biases[1],
                                  nonlinearity=nonlinearity, order=order[3:],
                                  inplace_nonlinearity=inplace_nonlinearity,
                                  stride=second_stride, blur=second_blur,
                                  **residual_params)
        # Shortcut branch.
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels, out_channels,
                                           bias=biases[2],
                                           nonlinearity=skip_nonlinearity_type,
                                           order=order[0:3],
                                           stride=shortcut_stride,
                                           blur=shortcut_blur,
                                           **shortcut_params)
        elif in_channels < out_channels:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels,
                                           out_channels - in_channels,
                                           bias=biases[2],
                                           nonlinearity=skip_nonlinearity_type,
                                           order=order[0:3],
                                           stride=shortcut_stride,
                                           blur=shortcut_blur,
                                           **shortcut_params)

        # Whether this block expects conditional inputs.
        self.conditional = getattr(self.conv_block_0, 'conditional', False) or \
                           getattr(self.conv_block_1, 'conditional', False)

    def _get_stride_blur(self):
        if self.stride > 1:
            # Downsampling.
            first_stride, second_stride = 1, self.stride
            first_blur, second_blur = False, self.blur
            shortcut_stride = self.stride
            shortcut_blur = self.blur
            self.upsample = None
        elif self.stride < 1:
            # Upsampling.
            first_stride, second_stride = self.stride, 1
            first_blur, second_blur = self.blur, False
            shortcut_blur = False
            shortcut_stride = 1
            if self.blur:
                # The shortcut branch uses blur_upsample + stride-1 conv
                self.upsample = BlurUpsample()
            else:
                shortcut_stride = self.stride
                self.upsample = nn.Upsample(scale_factor=2)
        else:
            first_stride = second_stride = 1
            first_blur = second_blur = False
            shortcut_stride = 1
            shortcut_blur = False
            self.upsample = None
        return (first_stride, second_stride, shortcut_stride,
                first_blur, second_blur, shortcut_blur)

    def conv_blocks(self, x, *cond_inputs, separate_cond=False,
                    **kw_cond_inputs):
        if separate_cond:
            dx = self.conv_block_0(x, cond_inputs[0],
                                   **kw_cond_inputs.get('kwargs_0', {}))
            dx = self.conv_block_1(dx, cond_inputs[1],
                                   **kw_cond_inputs.get('kwargs_1', {}))
        else:
            dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, separate_cond=False,
                **kw_cond_inputs):
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs,
                            separate_cond=separate_cond, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, separate_cond=separate_cond,
                                  **kw_cond_inputs)

        if self.upsample_first and self.upsample is not None:
            x = self.upsample(x)
        if self.learn_shortcut:
            if separate_cond:
                x_shortcut = self.conv_block_s(x, cond_inputs[2],
                                               **kw_cond_inputs.get('kwargs_2',
                                                                    {}))
            else:
                x_shortcut = self.conv_block_s(x, *cond_inputs,
                                               **kw_cond_inputs)
        elif self.in_channels < self.out_channels:
            if separate_cond:
                x_shortcut_pad = self.conv_block_s(x, cond_inputs[2],
                                                   **kw_cond_inputs.get(
                                                       'kwargs_2', {}))
            else:
                x_shortcut_pad = self.conv_block_s(x, *cond_inputs,
                                                   **kw_cond_inputs)
            x_shortcut = torch.cat((x, x_shortcut_pad), dim=1)
        elif self.in_channels > self.out_channels:
            x_shortcut = x[:, :self.out_channels, :, :]
        else:
            x_shortcut = x
        if not self.upsample_first and self.upsample is not None:
            x_shortcut = self.upsample(x_shortcut)

        output = x_shortcut + dx
        return self.output_scale * output

    def extra_repr(self):
        s = 'output_scale={output_scale}'
        return s.format(**self.__dict__)


class _BaseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params, activation_norm_type,
                 activation_norm_params, nonlinearity,
                 inplace_nonlinearity, apply_noise, blur, order, input_dim,
                 clamp, blur_kernel, output_scale,
                 init_gain):
        super().__init__()

        self.weight_norm_type = weight_norm_type
        self.stride = stride
        self.clamp = clamp
        self.init_gain = init_gain

        nonlinearity_layer = get_nonlinearity_layer(nonlinearity,
                                                    inplace=inplace_nonlinearity)

        # Noise injection layer.
        if not apply_noise: noise_layer = None
        # Convolutional layer.
        if blur:
            assert blur_kernel is not None
            if stride == 2:
                # Blur - Conv - Noise - Activate
                p = (len(blur_kernel) - 2) + (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2, p // 2
                padding = 0
                blur_layer = Blur(blur_kernel, pad=(pad0, pad1),
                                  padding_mode=padding_mode)
                order = order.replace('C', 'BC')
            elif stride == 0.5:
                # Conv - Blur - Noise - Activate
                padding = 0
                p = (len(blur_kernel) - 2) - (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2 + 1, p // 2 + 1
                blur_layer = Blur(blur_kernel, pad=(pad0, pad1),
                                  padding_mode=padding_mode)
                order = order.replace('C', 'CB')
            elif stride == 1:
                # No blur for now
                blur_layer = nn.Identity()
            else:
                raise NotImplementedError
        else:
            blur_layer = nn.Identity()

        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(weight_norm_type,
                                            **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, input_dim))

        # Normalization layer.
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(
            norm_channels, activation_norm_type, input_dim,
            **vars(activation_norm_params))

        # Mapping from operation names to layers.
        mappings = {'C': {'conv': conv_layer},
                    'N': {'norm': activation_norm_layer},
                    'A': {'nonlinearity': nonlinearity_layer}}
        mappings.update({'B': {'blur': blur_layer}})
        mappings.update({'G': {'noise': noise_layer}})

        # All layers in order.
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])

        # Whether this block expects conditional inputs.
        self.conditional = getattr(conv_layer, 'conditional', False) or \
                           getattr(activation_norm_layer, 'conditional', False)

        # Scale the output by a learnable scaler parameter.
        if output_scale is not None:
            self.output_scale = nn.Parameter(torch.tensor(output_scale))
        else:
            self.register_parameter("output_scale", None)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        for key, layer in self.layers.items():
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
            if self.clamp is not None and isinstance(layer, nn.Conv2d):
                x.clamp_(max=self.clamp)
            if key == 'conv':
                if self.output_scale is not None:
                    x = x * self.output_scale
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        # Returns the convolutional layer.
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            if stride < 1:  # Fractionally-strided convolution.
                padding_mode = 'zeros'
                assert padding == 0
                layer_type = getattr(nn, f'ConvTranspose{input_dim}d')
                stride = round(1 / stride)
            else:
                layer_type = getattr(nn, f'Conv{input_dim}d')
            layer = layer_type(in_channels, out_channels, kernel_size, stride,
                               padding, dilation=dilation,
                               groups=groups, bias=bias,
                               padding_mode=padding_mode)

        return layer


class Conv2dBlock(_BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', weight_norm_type='none',
                 weight_norm_params=None, activation_norm_type='none',
                 activation_norm_params=None, nonlinearity='none',
                 inplace_nonlinearity=False, apply_noise=False,
                 blur=False, order='CNA', clamp=None, blur_kernel=(1, 3, 3, 1),
                 output_scale=None, init_gain=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise, blur,
                         order, 2, clamp, blur_kernel,
                         output_scale, init_gain)


class LinearBlock(_BaseConvBlock):
    def __init__(self, in_features, out_features, bias=True,
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none',
                 inplace_nonlinearity=False, apply_noise=False, order='CNA',
                 clamp=None, blur_kernel=(1, 3, 3, 1),
                 output_scale=None, init_gain=1.0, **_kwargs):
        if bool(_kwargs):
            warnings.warn(f"Unused keyword arguments {_kwargs}")
        super().__init__(in_features, out_features, None, None, None, None,
                         None, bias, None, weight_norm_type,
                         weight_norm_params, activation_norm_type,
                         activation_norm_params, nonlinearity,
                         inplace_nonlinearity, apply_noise, False, order, 0,
                         clamp, blur_kernel, output_scale,
                         init_gain)


class Res2dBlock(_BaseResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', weight_norm_type='none',
                 weight_norm_params=None, activation_norm_type='none',
                 activation_norm_params=None, skip_activation_norm=True,
                 skip_nonlinearity=False, skip_weight_norm=True,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False,
                 hidden_channels_equal_out_channels=False, order='CNACNA',
                 learn_shortcut=None, clamp=None,
                 output_scale=1, blur=False, upsample_first=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity, nonlinearity,
                         inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels, order, Conv2dBlock,
                         learn_shortcut, clamp, output_scale,
                         blur=blur, upsample_first=upsample_first,
                         skip_weight_norm=skip_weight_norm)


class MunitGenerator(nn.Module):
    def __init__(self, gen_cfg={}):
        super().__init__()
        self.autoencoder_a = MunitAutoEncoder(**gen_cfg)
        self.autoencoder_b = MunitAutoEncoder(**gen_cfg)

    def forward(self, content_images, a2b=True, style_images=None,
                target_style=None, device=torch.device('cpu')):
        if a2b:
            content_encode = self.autoencoder_a.content_encoder
            style_encode = self.autoencoder_b.style_encoder
            decode = self.autoencoder_b.decode
        else:
            content_encode = self.autoencoder_b.content_encoder
            style_encode = self.autoencoder_a.style_encoder
            decode = self.autoencoder_a.decode

        content = content_encode(content_images)
        if style_images is not None:
            style = style_encode(style_images)
        elif target_style is not None:
            style = target_style
        else:
            style_channels = self.autoencoder_a.style_channels
            style = torch.randn(content.size(0), style_channels, 1, 1,
                                device=device)
        output_images = decode(content, style)

        return output_images


class MunitAutoEncoder(nn.Module):
    def __init__(self, num_filters=64, max_num_filters=256, num_filters_mlp=256,
                 latent_dim=8, num_res_blocks=4,
                 num_mlp_blocks=2, num_downsamples_style=4,
                 num_downsamples_content=2, num_image_channels=3,
                 content_norm_type='instance', style_norm_type='',
                 decoder_norm_type='instance', weight_norm_type='',
                 decoder_norm_params=SimpleNamespace(affine=False),
                 output_nonlinearity='', pre_act=True,
                 apply_noise=False, **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type': warnings.warn(
                "Generator argument '{}' is not used.".format(key))
        self.style_encoder = MunitStyleEncoder(num_downsamples_style,
                                               num_image_channels, num_filters,
                                               latent_dim,
                                               'reflect', style_norm_type,
                                               weight_norm_type, 'relu')
        self.content_encoder = MunitContentEncoder(num_downsamples_content,
                                                   num_res_blocks,
                                                   num_image_channels,
                                                   num_filters, max_num_filters,
                                                   'reflect', content_norm_type,
                                                   weight_norm_type, 'relu',
                                                   pre_act)
        self.decoder = MunitDecoder(num_downsamples_content, num_res_blocks,
                                    self.content_encoder.output_dim,
                                    num_image_channels, num_filters_mlp,
                                    'reflect', decoder_norm_type,
                                    decoder_norm_params, weight_norm_type,
                                    'relu', output_nonlinearity, pre_act,
                                    apply_noise)
        self.mlp = MunitMLP(latent_dim, num_filters_mlp, num_filters_mlp,
                            num_mlp_blocks, 'none', 'relu')
        self.style_channels = latent_dim

    def forward(self, images):
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        style = self.mlp(style)
        images = self.decoder(content, style)
        return images


class MunitContentEncoder(nn.Module):
    def __init__(self, num_downsamples, num_res_blocks, num_image_channels,
                 num_filters, max_num_filters, padding_mode,
                 activation_norm_type, weight_norm_type, nonlinearity,
                 pre_act=False):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity)
        # Whether or not it is safe to use inplace nonlinear activation.
        if not pre_act or (
                activation_norm_type != '' and activation_norm_type != 'none'):
            conv_params['inplace_nonlinearity'] = True

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        model = []
        model += [Conv2dBlock(num_image_channels, num_filters, 7, 1, 3,
                              **conv_params)]

        # Downsampling blocks.
        for i in range(num_downsamples):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model += [Conv2dBlock(num_filters_prev, num_filters, 4, 2, 1,
                                  **conv_params)]

        # Residual blocks.
        for _ in range(num_res_blocks):
            model += [Res2dBlock(num_filters, num_filters, **conv_params,
                                 order=order)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        return self.model(x)


class MunitStyleEncoder(nn.Module):
    def __init__(self, num_downsamples, num_image_channels, num_filters,
                 style_channels, padding_mode,
                 activation_norm_type, weight_norm_type, nonlinearity):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity, inplace_nonlinearity=True)
        model = []
        model += [Conv2dBlock(num_image_channels, num_filters, 7, 1, 3,
                              **conv_params)]
        for i in range(2):
            model += [Conv2dBlock(num_filters, 2 * num_filters, 4, 2, 1,
                                  **conv_params)]
            num_filters *= 2
        for i in range(num_downsamples - 2):
            model += [
                Conv2dBlock(num_filters, num_filters, 4, 2, 1, **conv_params)]
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(num_filters, style_channels, 1, 1, 0)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        return self.model(x)


class MunitDecoder(nn.Module):
    def __init__(self, num_upsamples, num_res_blocks, num_filters,
                 num_image_channels, style_channels, padding_mode,
                 activation_norm_type, activation_norm_params, weight_norm_type,
                 nonlinearity, output_nonlinearity,
                 pre_act=False, apply_noise=False):
        super().__init__()
        adain_params = SimpleNamespace(
            activation_norm_type=activation_norm_type,
            activation_norm_params=activation_norm_params,
            cond_dims=style_channels)
        conv_params = dict(padding_mode=padding_mode, nonlinearity=nonlinearity,
                           inplace_nonlinearity=True,
                           apply_noise=apply_noise,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type='adaptive',
                           activation_norm_params=adain_params)

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        # Residual blocks with AdaIN.
        self.decoder = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.decoder += [Res2dBlock(num_filters, num_filters, **conv_params,
                                        order=order)]

        # Convolutional blocks with upsampling.
        for i in range(num_upsamples):
            self.decoder += [nn.Upsample(scale_factor=2)]
            self.decoder += [Conv2dBlock(num_filters, num_filters // 2, 5, 1, 2,
                                         **conv_params)]
            num_filters //= 2
        self.decoder += [Conv2dBlock(num_filters, num_image_channels, 7, 1, 3,
                                     nonlinearity=output_nonlinearity,
                                     padding_mode=padding_mode)]

    def forward(self, x, style):
        for block in self.decoder:
            if getattr(block, 'conditional', False):
                x = block(x, style)
            else:
                x = block(x)
        return x


class MunitMLP(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, num_layers,
                 norm, nonlinearity):
        super().__init__()
        model = []
        model += [LinearBlock(input_dim, latent_dim, activation_norm_type=norm,
                              nonlinearity=nonlinearity)]
        for i in range(num_layers - 2):
            model += [
                LinearBlock(latent_dim, latent_dim, activation_norm_type=norm,
                            nonlinearity=nonlinearity)]
        model += [LinearBlock(latent_dim, output_dim, activation_norm_type=norm,
                              nonlinearity=nonlinearity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class UnitGenerator(nn.Module):
    r"""Improved UNIT generator.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
    """

    def __init__(self):
        super().__init__()
        self.autoencoder_a = UnitAutoEncoder()
        self.autoencoder_b = UnitAutoEncoder()

    def forward(self, content_images, a2b=True):
        if a2b:
            content_encode = self.autoencoder_a.content_encoder
            decode = self.autoencoder_b.decoder
        else:
            content_encode = self.autoencoder_b.content_encoder
            decode = self.autoencoder_a.decoder

        content = content_encode(content_images)
        output_images = decode(content)

        return output_images


class UnitAutoEncoder(nn.Module):
    def __init__(self,
                 num_filters=48,
                 max_num_filters=256,
                 num_res_blocks=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 content_norm_type='instance',
                 decoder_norm_type='instance',
                 weight_norm_type='',
                 output_nonlinearity='',
                 pre_act=True,
                 apply_noise=False,
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type':
                warnings.warn(
                    "Generator argument '{}' is not used.".format(key))
        self.content_encoder = UnitContentEncoder(num_downsamples_content,
                                                  num_res_blocks,
                                                  num_image_channels,
                                                  num_filters,
                                                  max_num_filters,
                                                  'reflect',
                                                  content_norm_type,
                                                  weight_norm_type,
                                                  'relu',
                                                  pre_act)
        self.decoder = UnitDecoder(num_downsamples_content,
                                   num_res_blocks,
                                   self.content_encoder.output_dim,
                                   num_image_channels,
                                   'reflect',
                                   decoder_norm_type,
                                   weight_norm_type,
                                   'relu',
                                   output_nonlinearity,
                                   pre_act,
                                   apply_noise)

    def forward(self, images):
        content = self.content_encoder(images)
        images_recon = self.decoder(content)
        return images_recon


class UnitContentEncoder(nn.Module):
    def __init__(self,
                 num_downsamples,
                 num_res_blocks,
                 num_image_channels,
                 num_filters,
                 max_num_filters,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity,
                 pre_act=False):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity)
        # Whether or not it is safe to use inplace nonlinear activation.
        if not pre_act or (activation_norm_type != '' and
                           activation_norm_type != 'none'):
            conv_params['inplace_nonlinearity'] = True

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        model = []
        model += [Conv2dBlock(num_image_channels, num_filters, 7, 1, 3,
                              **conv_params)]

        # Downsampling blocks.
        for i in range(num_downsamples):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model += [Conv2dBlock(num_filters_prev, num_filters, 4, 2, 1,
                                  **conv_params)]

        # Residual blocks.
        for _ in range(num_res_blocks):
            model += [Res2dBlock(num_filters, num_filters,
                                 **conv_params,
                                 order=order)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x)


class UnitDecoder(nn.Module):
    def __init__(self,
                 num_upsamples,
                 num_res_blocks,
                 num_filters,
                 num_image_channels,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity,
                 output_nonlinearity,
                 pre_act=False,
                 apply_noise=False):
        super().__init__()

        conv_params = dict(padding_mode=padding_mode,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True,
                           apply_noise=apply_noise,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type=activation_norm_type)

        # The order of operations in residual blocks.
        order = 'pre_act' if pre_act else 'CNACNA'

        # Residual blocks.
        self.decoder = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.decoder += [Res2dBlock(num_filters, num_filters,
                                        **conv_params,
                                        order=order)]

        # Convolutional blocks with upsampling.
        for i in range(num_upsamples):
            self.decoder += [nn.Upsample(scale_factor=2)]
            self.decoder += [Conv2dBlock(num_filters, num_filters // 2,
                                         5, 1, 2, **conv_params)]
            num_filters //= 2
        self.decoder += [Conv2dBlock(num_filters, num_image_channels, 7, 1, 3,
                                     nonlinearity=output_nonlinearity,
                                     padding_mode=padding_mode)]

    def forward(self, x):
        for block in self.decoder:
            x = block(x)
        return x
