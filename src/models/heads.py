import torch
from torch import nn
import torch.nn.functional as F

from .models import UNet, LightWeight


class UNetDiscriminator(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNetDiscriminator, self).__init__()
        self.init_layer = nn.MaxPool2d(2)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1))
        self.fullyConnected = nn.Sequential(nn.Flatten(start_dim=1),
                                            nn.Linear(256 * 6 * 6, 4096),
                                            nn.ReLU(), nn.Dropout(),
                                            nn.Linear(4096, 1024), nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(1024, num_classes))

    def forward(self, d1, z_up=None):
        output = self.init_layer(d1)
        if z_up is not None:
            output += z_up
        output = self.conv_layers(output)
        output = self.fullyConnected(output)

        return output
