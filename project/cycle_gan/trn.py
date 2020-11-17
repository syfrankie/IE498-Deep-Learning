import numpy as np
import itertools
import random
import os
import time
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

from gen_model import Generator
from dis_model import Discriminator
from function import ImageDataset
from function import ReplayBuffer
from function import LambdaLR


lr = 0.0002
num_epochs = 160
batch_size = 1
start_epoch = 0
decay_epoch = 150
lambda_cycle = 10
lambda_iden = 0


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
trn_set = ImageDataset('../dataset/horse2zebra/', transforms_=transforms_trn, mode='train')
trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True, num_workers=8)


# gen & dis
G, F = Generator(num_blocks=9), Generator(num_blocks=9)
D_X, D_Y = Discriminator(), Discriminator()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G.to(device)
F.to(device)
D_X.to(device)
D_Y.to(device)

G.load_state_dict(torch.load('output/G.pth'))
F.load_state_dict(torch.load('output/F.pth'))
D_X.load_state_dict(torch.load('output/D_X.pth'))
D_Y.load_state_dict(torch.load('output/D_Y.pth'))

# weights
'''
G.apply(initial_weights)
F.apply(initial_weights)
D_X.apply(initial_weights)
D_Y.apply(initial_weights)
'''

#loss
GAN_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
iden_loss = nn.L1Loss()

# optim
optimizer_G_F = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_X = optim.Adam(D_X.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=lr, betas=(0.5, 0.999))

# lr scheduler
lr_scheduler_G_F = torch.optim.lr_scheduler.LambdaLR(optimizer_G_F, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_epoch).step)

'''
def Loss1(input, net, loss, target):
    output = net(input)
    return loss(output, target)

def Loss2(input, net1, net2, loss, target):
    output1 = net1(input)
    output2 = net2(output1)
    return loss(output2, target)
'''

Tensor = torch.cuda.FloatTensor
#Tensor = torch.Tensor
fake_Xs, fake_Ys = ReplayBuffer(), ReplayBuffer()



for epoch in range(130, num_epochs):
    start = time.time()
    Loss_X = 0
    Loss_Y = 0
    Loss_GF = 0

    for idx, data in enumerate(trn_loader):

        #X = data['X'].to(device)
        #Y = data['Y'].to(device)
        X = Variable(Tensor(batch_size, 3, 256, 256).copy_(data['X']))
        Y = Variable(Tensor(batch_size, 3, 256, 256).copy_(data['Y']))


        # loss of G & F
        optimizer_G_F.zero_grad()
        for param in G.parameters():
            param.requires_grad = True
        for param in F.parameters():
            param.requires_grad = True
        for param in D_X.parameters():
            param.requires_grad = False
        for param in D_Y.parameters():
            param.requires_grad = False

        G_X, F_Y = G(X), F(Y)
        F_X, G_Y = F(X), G(Y)
        F_G_X, G_F_Y = F(G_X), G(F_Y)
        D_Y_X, D_X_Y = D_Y(G_X), D_X(F_Y)
        
        Loss_Iden = iden_loss(F_X, X) + iden_loss(G_Y, Y)
        Loss_Cycle = cycle_loss(F_G_X, X) + cycle_loss(G_F_Y, Y)
        Loss_GAN = GAN_loss(D_Y_X, Tensor(batch_size).fill_(1.0)) + GAN_loss(D_X_Y, Tensor(batch_size).fill_(1.0))
        #Loss_Iden = (Loss1(X, F, iden_loss, X)+Loss1(Y, G, iden_loss, X)) * lambda_iden
        #Loss_Cycle = (Loss2(X, G, F, cycle_loss, X)+Loss2(Y, F, G, cycle_loss, Y)) * lambda_cycle
        #Loss_GAN = Loss2(X, G, D_Y, GAN_loss, Tensor(batch_size).fill_(1.0)) + Loss2(Y, F, D_X, GAN_loss, Tensor(batch_size).fill_(1.0))

        Loss_G_F = lambda_iden*Loss_Iden + lambda_cycle*Loss_Cycle + Loss_GAN
        Loss_G_F.backward()
        for group in optimizer_G_F.param_groups:
            for p in group['params']:
                state = optimizer_G_F.state[p]
                if 'step' in state.keys():
                    if ('step' in state and state['step']>=1022):
                        state['step'] = 1000
        optimizer_G_F.step()


        # loss of D_X
        for param in G.parameters():
            param.requires_grad = False
        for param in F.parameters():
            param.requires_grad = False
        for param in D_X.parameters():
            param.requires_grad = True

        optimizer_D_X.zero_grad()
        Loss_Real = GAN_loss(D_X(X), Tensor(batch_size).fill_(1.0))
        #Loss_Real = Loss1(X, D_X, GAN_loss, Tensor(batch_size).fill_(1.0))
        fake_X = fake_Xs.add_sample(F(Y), batch_size)
        Loss_Fake = GAN_loss(D_X(fake_X.detach()), Tensor(batch_size).fill_(0.0))

        Loss_D_X = .5 * (Loss_Real + Loss_Fake)
        Loss_D_X.backward()
        for group in optimizer_D_X.param_groups:
            for p in group['params']:
                state = optimizer_D_X.state[p]
                if 'step' in state.keys():
                    if ('step' in state and state['step']>=1022):
                        state['step'] = 1000
        optimizer_D_X.step()


        # loss of D_Y
        for param in D_X.parameters():
            param.requires_grad = False
        for param in D_Y.parameters():
            param.requires_grad = True

        optimizer_D_Y.zero_grad()
        Loss_Real = GAN_loss(D_Y(Y), Tensor(batch_size).fill_(1.0))
        #Loss_Real = Loss1(Y, D_Y, GAN_loss, Tensor(batch_size).fill_(1.0))
        fake_Y = fake_Ys.add_sample(G(X), batch_size)
        Loss_Fake = GAN_loss(D_Y(fake_Y.detach()), Tensor(batch_size).fill_(0.0))

        Loss_D_Y = .5 * (Loss_Real + Loss_Fake)
        Loss_D_Y.backward()
        for group in optimizer_D_Y.param_groups:
            for p in group['params']:
                state = optimizer_D_Y.state[p]
                if 'step' in state.keys():
                    if ('step' in state and state['step']>=1022):
                        state['step'] = 1000
        optimizer_D_Y.step()


        Loss_GF += Loss_G_F.item()
        Loss_X += Loss_D_X.item()
        Loss_Y += Loss_D_Y.item()

        Loss_GF /= len(trn_loader)
        Loss_X /= len(trn_loader)
        Loss_Y /= len(trn_loader)

    lr_scheduler_G_F.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()
    print('Epoch %d | Cost %.1f sec' % (epoch + 1, time.time() - start))
    print('Loss_D_X %.4f | Loss_D_Y %.4f | Loss_G_F %.4f' % (Loss_X, Loss_Y, Loss_GF))

    torch.save(G.state_dict(), 'output/G.pth')
    torch.save(F.state_dict(), 'output/F.pth')
    torch.save(D_X.state_dict(), 'output/D_X.pth')
    torch.save(D_Y.state_dict(), 'output/D_Y.pth')

    if (Loss_GF < 0.002): break






