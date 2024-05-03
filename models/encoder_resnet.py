import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import resnet18

import numpy as np


class ResNet18(nn.Module):
    def __init__(self, arch='resnet18'):
        super(ResNet18, self).__init__()
        self.arch = arch
        if self.arch == 'resnet18':
            self.encoder = resnet18(False)
        else:
            raise Exception("Invalid arch name {}".format(self.args.dataset))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_tmp = None
        
    def forward(self, x):
        x             = self.encoder(x)
        x             = self.avgpool(x).squeeze(-1).squeeze(-1)
        self.feat_tmp = x
        return x


