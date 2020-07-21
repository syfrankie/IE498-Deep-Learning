
# coding: utf-8


import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #input shape 3*32*32
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1) #32*32*32
        self.batnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.maxpool1 = nn.MaxPool2d(2, 2) #32*16*16
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 64, 3, padding = 1) #64*16*16
        self.batnorm2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(2, 2) #64*8*8
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(64, 64, 3, padding = 1) #64*8*8
        self.batnorm3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3) #64*6*6
        self.dropout3 = nn.Dropout(0.25)
        
        self.conv7 = nn.Conv2d(64, 128, 3, padding = 1) #128*6*6
        self.batnorm4 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3) #128*4*4
        self.batnorm5 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128*4*4, 500) #1*500
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10) #1*10
        

    def forward(self, x):
        x = self.batnorm1(F.relu(self.conv1(x)))
        x = self.dropout1(self.maxpool1(F.relu(self.conv2(x))))

        x = self.batnorm2(F.relu(self.conv3(x)))
        x = self.dropout2(self.maxpool2(F.relu(self.conv4(x))))

        x = self.batnorm3(F.relu(self.conv5(x)))
        x = self.dropout3(F.relu(self.conv6(x)))

        x = self.batnorm4(F.relu(self.conv7(x)))
        x = self.dropout4(self.batnorm5(F.relu(self.conv8(x))))
        
        x = x.view(-1, 128*4*4)
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

net = Net()

alpha = 0.0005
batch_size = 40
num_epochs = 10

import torch.optim as optim
import time
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(),lr=alpha)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

import h5py
cifar10 = h5py.File('CIFAR10.hdf5','r')
list(cifar10.keys())
trn_x_old,trn_y = np.float32(cifar10['X_train'][:]),np.int32(cifar10['Y_train'][:])
tst_x,tst_y = np.float32(cifar10['X_test'][:]),np.int32(cifar10['Y_test'][:])

index = np.random.permutation(np.shape(trn_y)[0])
trn_x_old = trn_x_old[index,:]
trn_y = trn_y[index]
trn_x_ori = torch.from_numpy(trn_x_old)

img_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1)
            ])
trn_x = torch.zeros(size=np.shape(trn_x_old))
for n in range(np.shape(trn_x_old)[0]):
    img = transforms.functional.to_pil_image(trn_x_ori[n,:])
    img = img_aug(img)
    trn_x[n,:] = transforms.functional.to_tensor(img)


for epoch in range(num_epochs):
    start = time.time()
    net.train()
    for i in range(0, np.shape(trn_y)[0], batch_size):
        trn_x_batch = torch.FloatTensor( trn_x[i:i+batch_size,:] )
        trn_y_batch = torch.LongTensor( trn_y[i:i+batch_size] )
        #image, label = Variable(trn_x_batch), Variable(trn_y_batch)
        image, label = Variable(trn_x_batch).cuda(), Variable(trn_y_batch).cuda()

        optimizer.zero_grad()
        output = net(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print('epoch %d cost %3f sec loss %.4f' % (epoch + 1, time.time() - start, loss.item()))
print('Finished')


counter1 = 0
train_accuracy_sum = 0.0
net.eval()
for i in range(0, 10000, batch_size):
    x_train_batch = torch.FloatTensor( trn_x[i:i+batch_size,:] )
    y_train_batch = torch.LongTensor( trn_y[i:i+batch_size] )
    #image, label = Variable(x_train_batch), Variable(y_train_batch)
    image, label = Variable(x_train_batch).cuda(), Variable(y_train_batch).cuda()
    output1 = net(image)
    prediction1 = output1.data.max(1)[1]
    accuracy1 = ( float( prediction1.eq(label.data).sum() ) /float(batch_size)  )*100.0
    counter1 += 1
    train_accuracy_sum = train_accuracy_sum + accuracy1
train_accuracy_ave = train_accuracy_sum/float(counter1)
print('Training Accuracy is:',train_accuracy_ave)

counter2 = 0
test_accuracy_sum = 0.0
net.eval()
for i in range(0, np.shape(tst_y)[0], batch_size):
    x_test_batch = torch.FloatTensor( tst_x[i:i+batch_size,:] )
    y_test_batch = torch.LongTensor( tst_y[i:i+batch_size] )
    #image, label = Variable(x_test_batch), Variable(y_test_batch)
    image, label = Variable(x_test_batch).cuda(), Variable(y_test_batch).cuda()
    output2 = net(image)
    prediction2 = output2.data.max(1)[1]
    accuracy2 = ( float( prediction2.eq(label.data).sum() ) /float(batch_size)  )*100.0
    counter2 += 1
    test_accuracy_sum = test_accuracy_sum + accuracy2
test_accuracy_ave = test_accuracy_sum/float(counter2)
print('Testing Accuracy is:',test_accuracy_ave)
