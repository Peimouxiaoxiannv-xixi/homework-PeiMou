import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

class Conv2dBNLeakyRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBNLeakyRelu, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),

            nn.LeakyReLU(1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBNLeakyRelu(nchannels, int(nchannels/2), 1, 1),
            Conv2dBNLeakyRelu(int(nchannels/2), nchannels, 3, 1),
        )

    def forward(self, x):
        return x + self.block(x)

class TcwNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        layer_list = []
        layer_list.append(OrderedDict([

        ]))
        self.conv = nn.Sequential(
            Conv2dBNLeakyRelu(3, 32, 3, 1),
            Conv2dBNLeakyRelu(32, 64, 3, 2),
            ResBlock(64),
            Conv2dBNLeakyRelu(64, 128, 3, 2),
            ResBlock(128),
            ResBlock(128),
            Conv2dBNLeakyRelu(128, 256, 3, 2),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            Conv2dBNLeakyRelu(256, 512, 3, 2),
        )

        self.fn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512*14*14, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fn(x)
        return x

