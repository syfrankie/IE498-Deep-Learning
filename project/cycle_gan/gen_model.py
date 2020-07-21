import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Res_Block(nn.Module):
    def __init__(self, in_channels):
        super(Res_Block, self).__init__()
        self.res_fun = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    def forward(self,x):
        x += self.res_fun(x)
        return x

class Generator(nn.Module):
    def __init__(self, num_blocks):
        super(Generator, self).__init__()
        # input 256x256x3
        gen_channel = 64
        gen_kernel = 3
        # input block
        self.input = nn.Sequential(
            nn.ReflectionPad2d(3), # 262x262x3
            nn.Conv2d(3, gen_channel, 7), # 256x256x64
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # downsample block
        self.downsample = nn.Sequential(
            nn.Conv2d(gen_channel, gen_channel*2, gen_kernel, stride=2, padding=1), # 128x128x128
            nn.InstanceNorm2d(gen_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(gen_channel*2, gen_channel*4, gen_kernel, stride=2, padding=1), # 64x64x256
            nn.InstanceNorm2d(gen_channel*4),
            nn.ReLU(inplace=True)
        )

        # residual block
        layers = []
        for _ in range(num_blocks):
            layers.append(Res_Block(gen_channel*4))
        self.residual = nn.Sequential(*layers)

        # upsample block
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(gen_channel*4, gen_channel*2, gen_kernel, stride=2, padding=1), # 128x128x128
            nn.InstanceNorm2d(gen_channel*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(gen_channel*2, gen_channel, gen_kernel, stride=2, padding=1), # 256x256x64
            nn.InstanceNorm2d(gen_channel*2),
            nn.ReLU(inplace=True)
        )

        # output block
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3), # 262x262x64
            nn.Conv2d(gen_channel, 3, 7), # 256x256x3
            nn.Tanh()
        )

    def forward(self,x):
        x = self.input(x)
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        x = self.output(x)
        return x

        
