import sys
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
tst_set = ImageDataset('../datasets/horse2zebra/', transforms_=transforms_tst, mode='test')
tst_loader = DataLoader(tst_set, batch_size=batch_size, shuffle=False, num_workers=8)


G, F = Generator(num_blocks=9), Generator(num_blocks=9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G.to(device)
F.to(device)
G.load_state_dict(torch.load('output/G.pth'))
F.load_state_dict(torch.load('output/F.pth'))
G.eval
F.eval




