import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AlexNet(nn.Module):
    def __init__(self, cls=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 数据集较小，输出96设置为原论文一半48
            # 在左右上下各补两行0
            # 224*224*3
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # 55*55*48
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            # 27*27*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            # 13*13*192
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            # 13*13*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 6*6*128
        )

        self.classifier = nn.Sequential(
            # 防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cls),
        )
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
            return x

    def initialize_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                # mod为卷积层使用凯明normal进行权重初始化
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
                elif isinstance(mod, nn.Linear):
                    # 对全连接层进行正态分布赋值，均值为0，方差为0.01
                    nn.init.normal_(mod.weight, 0, 0.01)
                    # 偏置初始化为0
                    nn.init.constant_(mod.bias, 0)

