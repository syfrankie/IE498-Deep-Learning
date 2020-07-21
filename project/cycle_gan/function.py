import numpy as np
import random
import glob
import os
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_X = sorted(glob.glob(os.path.join(root, '%s/X' % mode) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(root, '%s/Y' % mode) + '/*.*'))

    def __getitem__(self, index):
        image_X = self.transform(Image.open(self.files_X[index % len(self.files_X)]))

        if self.unaligned:
            image_Y = self.transform(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        else:
            image_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))

        return {"X": image_X, "Y": image_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))


class ReplayBuffer(object):
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.buffer = []

    def add_sample(self, img):
        to_return = []
        for idx in img.buffer:
            idx = torch.unsqueeze(idx, 0)
            if len(self.buffer) < self.capacity:
                self.buffer.append(idx)
                to_return.append(idx)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.capacity-1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = idx
                else:
                    to_return.append(idx)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)







