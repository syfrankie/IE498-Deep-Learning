import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # input 256x256x3
        dis_channel = 64
        dis_kernel = 4
        self.discrim = nn.Sequential(
            nn.Conv2d(3, dis_channel, dis_kernel, stride=2, padding=1), # 128x128x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dis_channel, dis_channel*2, dis_kernel, stride=2, padding=1), # 64x64x128
            nn.InstanceNorm2d(dis_channel*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dis_channel*2, dis_channel*4, dis_kernel, stride=2, padding=1), # 32x32x256
            nn.InstanceNorm2d(dis_channel*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dis_channel*4, dis_channel*8, dis_kernel, padding=1), # 31x31x512
            nn.InstanceNorm2d(dis_channel*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dis_channel*8, 1, dis_kernel, padding=1), # 30x30x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self,x):
        x = self.discrim(x)
        x = x.view(x.size()[0], -1)
        return x