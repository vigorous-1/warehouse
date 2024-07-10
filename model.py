# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
#               ★★★请在空白处填写适当的语句，将CustomNet类的定义补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    """自定义神经网络模型。"""

    def __init__(self):
        """初始化方法。在本方法中，请完成神经网络的各个模块/层的定义。请确保每层的输出维度与下一层的输入维度匹配。"""
        super(CustomNet, self).__init__()

        # 定义第一个卷积层 + BatchNorm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 定义第二个卷积层 + BatchNorm
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 定义第三个卷积层 + BatchNorm
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 定义第四个卷积层 + BatchNorm
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 定义全连接层输入的特征向量的大小
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        """前向传播方法。定义数据如何通过各层处理。"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 256 * 4 * 4)  # 展开特征图，使其适配全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('./images/train.txt', './images/train', ToTensor)
    net = CustomNet()                                # 实例化
    print(net)
    print(type(c))
    x = torch.unsqueeze(c[10][0], 0)      # 模拟一个模型的输入数据
    print(net.forward(x))                            # 测试forward方法
