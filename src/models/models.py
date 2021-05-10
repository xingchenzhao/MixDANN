import torch.nn.functional as F
import numpy as np
import torch


def init_conv_weights(m, activations='relu'):

    gain = torch.nn.init.calculate_gain(activations)

    if type(m) == torch.nn.Conv2d  \
            or type(m) == torch.nn.ConvTranspose2d:

        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0.0)


class LightWeight(torch.nn.Module):
    def __init__(self, num_filters=64):
        super().__init__()

        nf = num_filters

        in_channels = 1
        out_channels = 1

        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, nf, 3), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(nf), torch.nn.Conv2d(nf, 2 * nf, 3),
            torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(2 * nf),
            torch.nn.Conv2d(2 * nf, 2 * nf, 3), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(2 * nf))

        self.conv_up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2 * nf, 2 * nf, 3),
            torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(2 * nf),
            torch.nn.ConvTranspose2d(2 * nf, nf, 3),
            torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(nf),
            torch.nn.ConvTranspose2d(nf, out_channels, 3))

        self.apply(init_conv_weights)

    def forward(self, x):
        h = self.conv_down(x)
        return self.conv_up(h)


class UNet(torch.nn.Module):
    """
    Adapted from work of github user milesial
    Available at: https://github.com/milesial/Pytorch-UNet
    Original Paper: https://arxiv.org/abs/1505.04597
    See UNET-LICENSE in license directory
    """
    def __init__(self, bilinear=False, T1=False):
        super().__init__()
        if T1:
            self.input_conv = self._double_conv(
                2, 64)  # new experiment kernel size 5
        else:
            self.input_conv = self._double_conv(1, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 512)
        self.up1 = self._up(1024, 256, bilinear)
        self.up2 = self._up(512, 128, bilinear)
        self.up3 = self._up(256, 64, bilinear)
        self.up4 = self._up(128, 64, bilinear)
        self.output_conv = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.apply(init_conv_weights)

    def forward(self, x):
        h = self.input_conv(x)
        d1 = self.down1(h)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1((d4, d3))
        u2 = self.up2((u1, d2))
        u3 = self.up3((u2, d1))
        u4 = self.up4((u3, h))
        output = self.output_conv(u4)
        return output, d1, u2

    def _double_conv(self, in_channels, out_channels, kernel_size=3):
        ic = in_channels
        oc = out_channels
        return torch.nn.Sequential(
            torch.nn.Conv2d(ic, oc, kernel_size, padding=1),
            torch.nn.BatchNorm2d(oc),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(oc, oc, kernel_size, padding=1),
            torch.nn.BatchNorm2d(oc),
            torch.nn.ReLU(inplace=True),
        )

    def _down(self, in_channels, out_channels):
        ic = in_channels
        oc = out_channels
        return torch.nn.Sequential(torch.nn.MaxPool2d(2),
                                   self._double_conv(ic, oc))

    def _up(self, in_channels, out_channels, bilinear):
        ic = in_channels
        oc = out_channels
        return torch.nn.Sequential(
            UNet.Up(in_channels, out_channels, bilinear=bilinear),
            self._double_conv(ic, oc))

    class Up(torch.nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):

            ic = in_channels
            oc = out_channels
            super().__init__()

            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = torch.nn.Upsample(scale_factor=2,
                                            mode='bilinear',
                                            align_corners=True)
            else:
                self.up = torch.nn.ConvTranspose2d(ic // 2,
                                                   ic // 2,
                                                   kernel_size=2,
                                                   stride=2)

            # should be covered by parent module, but in case of
            # outside use, it does not hurt to initialize twice
            self.apply(init_conv_weights)

        def forward(self, x):
            x1, x2 = x
            x1 = self.up(x1)
            # bxcxhxw
            h_diff = x2.size()[2] - x1.size()[2]
            w_diff = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (w_diff // 2, w_diff - w_diff // 2, h_diff // 2,
                            h_diff - h_diff // 2))
            return torch.cat([x2, x1], dim=1)


MODELS = {'unet': UNet, 'light-weight': LightWeight}
