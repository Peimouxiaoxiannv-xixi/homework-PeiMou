import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class Conv_dw(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Conv_dw, self).__init__()
        layers = []
        layers.append(ConvBNReLU(in_channel=in_channel, out_channel=in_channel, stride=stride, groups=in_channel))
        layers.append(ConvBNReLU(in_channel, out_channel, kernel_size=1, stride=1))
        self.con_dw = nn.Sequential(*layers)

    def forward(self, x):
        return self.con_dw(x)

class MobileNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1, self).__init__()
        input_channel = 32
        features = []
        features.append(ConvBNReLU(in_channel=3, out_channel=input_channel, stride=2))
        features.append(Conv_dw(input_channel, input_channel*2))
        input_channel = input_channel*2
        features.append(Conv_dw(input_channel, input_channel*2, 2))
        input_channel = input_channel*2
        for i in range(2):
            features.append(Conv_dw(input_channel, input_channel))
            features.append(Conv_dw(input_channel, input_channel*2, 2))
            input_channel = input_channel*2
        for i in range(5):
            features.append(Conv_dw(input_channel, input_channel))
        features.append(Conv_dw(input_channel, input_channel*2, 2))
        input_channel = input_channel*2
        features.append((Conv_dw(input_channel, input_channel, 2)))
        self.feature = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x





