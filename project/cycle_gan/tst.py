import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from gen_model import Generator
from dis_model import Discriminator
from function import ImageDataset
from function import ReplayBuffer
from function import LambdaLR

batch_size = 1


transforms_tst = [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
tst_set = ImageDataset('../dataset/horse2zebra/', transforms_=transforms_tst, mode='test')
tst_loader = DataLoader(tst_set, batch_size=batch_size, shuffle=False, num_workers=8)


G, F = Generator(num_blocks=9), Generator(num_blocks=9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G.to(device)
F.to(device)
G.load_state_dict(torch.load('output/G.pth'))
F.load_state_dict(torch.load('output/F.pth'))
G.eval
F.eval

'''
if not os.path.exists('output/X'):
    os.makedirs('output/X')
if not os.path.exists('output/Y'):
    os.makedirs('output/Y')
'''

Tensor = torch.cuda.FloatTensor
#Tensor = torch.Tensor

for idx, data in enumerate(tst_loader):

    X = Variable(Tensor(batch_size, 3, 256, 256).copy_(data['X']))
    Y = Variable(Tensor(batch_size, 3, 256, 256).copy_(data['Y']))
    G_X, F_Y = G(X), F(Y)
    F_G_X, G_F_Y = F(G_X), G(F_Y)

    origin_X = 0.5 * (X.data + 1.0)
    origin_Y = 0.5 * (Y.data + 1.0)
    fake_X= 0.5 * (G_X.data + 1.0)
    fake_Y = 0.5 * (F_Y.data + 1.0)
    cycle_X = 0.5 * (F_G_X.data + 1.0)
    cycle_Y = 0.5 * (G_F_Y.data + 1.0)

    save_image(origin_X, 'output/X/%04dorigin.png' % (idx+1))
    save_image(origin_Y, 'output/Y/%04dorigin.png' % (idx+1))
    save_image(fake_X, 'output/X/%04dfake.png' % (idx+1))
    save_image(fake_Y, 'output/Y/%04dfake.png' % (idx+1))
    save_image(cycle_X, 'output/X/%04dcycle.png' % (idx+1))
    save_image(cycle_Y, 'output/Y/%04dcycle.png' % (idx+1))



