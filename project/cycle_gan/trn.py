import numpy as np
import itertools
import glob
import random
import os
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from gan_model import Generator
from gan_model import Discriminator
from function import ImageDataset
from function import ReplayBuffer
from function import LambdaLR


lr = 0.0002
num_epochs = 2
batch_size = 1
start_epoch = 0
decay_epoch = 100
lambda_cycle = 10
lambda_iden = 5


def initial_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


transforms_trn = [ transforms.Resize(int(256*1.12), Image.BICUBIC), 
                transforms.RandomCrop(256), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
trn_set = ImageDataset('../datasets/horse2zebra/', transforms_=transforms_trn, mode='train')
trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True, num_workers=8)


#loss
GAN_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
iden_loss = nn.L1Loss()

# gen & dis
G, F = Generator(), Generator()
D_X, D_Y = Discriminator(), Discriminator()

# weights
G.apply(initial_weights)
F.apply(initial_weights)
D_X.apply(initial_weights)
D_Y.apply(initial_weights)

# optim
optimizer_G_F = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_X = optim.Adam(D_X.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=lr, betas=(0.5, 0.999))

# lr scheduler
lr_scheduler_G_F = torch.optim.lr_scheduler.LambdaLR(optimizer_G_F, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_epoch).step)


def Loss1(input, net, loss, target):
    output = net(input)
    return loss(output, target)

def Loss2(input, net1, net2, loss, target):
    output1 = net1(input)
    output2 = net2(output1)
    return loss(output2, target)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G.to(device)
F.to(device)
D_X.to(device)
D_Y.to(device)

Tensor = torch.cuda.FloatTensor
#Tensor = torch.Tensor



for epoch in range(num_epochs):
    for idx, data in enumerate(trn_loader):

        X = data['X'].to(device)
        Y = data['Y'].to(device)
        fake_Xs = ReplayBuffer()
        fake_Ys = ReplayBuffer()
        real_target = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
        fake_target = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

        # loss of G & F
        optimizer_G_F.zero_grad()
        Loss_Iden = (Loss1(X, F, iden_loss, X)+Loss1(Y, G, iden_loss, X)) * lambda_iden
        Loss_Cycle = (Loss2(X, G, F, cycle_loss, X)+Loss2(Y, F, G, cycle_loss, Y)) * lambda_cycle
        Loss_GAN = Loss2(X, G, D_Y, GAN_loss,real_target) + Loss2(Y, F, D_X, GAN_loss, real_target)
        Loss_G_F = Loss_Iden + Loss_Cycle + Loss_GAN
        Loss_G_F.backward()
        optimizer_G_F.step()

        # loss of D_X
        optimizer_D_X.zero_grad()
        Loss_Real = Loss1(X, D_X, GAN_loss, real_target)
        fake_X = fake_Xs.add_sample(F(Y))
        Loss_Fake = GAN_loss(D_X(fake_X.detach()), fake_target)
        Loss_D_X = .5 * (Loss_Real + Loss_Fake)
        Loss_D_X.backward()
        optimizer_D_X.step()

        # loss of D_Y
        optimizer_D_Y.zero_grad()
        Loss_Real = Loss1(Y, D_Y, GAN_loss, real_target)
        fake_Y = fake_Ys.add_sample(G(X))
        Loss_Fake = GAN_loss(D_Y(fake_Y.detach()), fake_target)
        Loss_D_Y = .5 * (Loss_Real + Loss_Fake)
        Loss_D_Y.backward()
        optimizer_D_Y.step()

    lr_scheduler_G_F.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()

    torch.save(G.state_dict(), 'output/G.pth')
    torch.save(F.state_dict(), 'output/F.pth')
    torch.save(D_X.state_dict(), 'output/D_X.pth')
    torch.save(D_Y.state_dict(), 'output/D_Y.pth')