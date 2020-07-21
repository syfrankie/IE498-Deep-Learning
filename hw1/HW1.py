
# coding: utf-8


#IE498 HW1
#Name: Yifan Shi(yifans16)
#Jan.30.2020

import numpy as np
import pandas as pd
import h5py
mnist = h5py.File('MNISTdata_1.hdf5','r')
list(mnist.keys())
trn_x,trn_y = np.array(mnist["x_train"][:]),np.array(mnist["y_train"][:])
tst_x,tst_y = np.array(mnist["x_test"][:]),np.array(mnist["y_test"][:])
# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# sigmoid prime
def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))
# softmax function 
def softmax(x):
    return np.divide(np.exp(x),np.sum(np.exp(x),axis=0))
# ReLU function
def ReLU(x):
    return np.maximum(0,x)
# ReLU prime
def ReLU_p(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
# unit number in the hidden layer
units = 100
# unit number in the output layer
k = 10
# size of mini-batch
size = 150

# initialize the parameters and LR
np.random.seed(11)
w = np.random.normal(0,.05,size=(units,np.shape(trn_x)[1]))
b1 = np.zeros(shape=(units,1))
# v1 is the momentum of weight w
v1 = np.zeros_like(w)
c = np.random.normal(0,.2,size=(k,units))
b2 = np.zeros(shape=(k,1))
# v2 is the momentum of weight c
v2 = np.zeros_like(c)
alpha = 0.0005
beta = 0.9

for l in range(10000):
    
    # update LR for each iteration
    #alpha = 0.01/(l+1)
    #alpha = 0.1*np.exp(-0.1*l)
    
    # randomly select the index
    i = np.sort(np.random.choice(trn_x.shape[0],size=size,replace=False))
    # transform b1 and b2 to change dim of bias
    b10 = np.matmul(b1,np.ones(shape=(1,size)))
    b20 = np.matmul(b2,np.ones(shape=(1,size)))
    # forward
    x = trn_x[[i]]
    z = np.add(np.matmul(w,x.T),b10)
    u = np.add(np.matmul(c,ReLU(z)),b20)
    #rho = -np.log(softmax(u))
    
    # backward
    # define the true y
    y = np.zeros(shape=(k,size))
    for s in range(size):
        y[trn_y[i[s]],s] = 1
    # partial derivative of rho on u
    rho_p = -np.subtract(y,softmax(u))
    delta = np.matmul(c.T,rho_p)
    # update parameters
    # new momentum = beta*old momentum - LR*GradientLoss
    # new weight = old weight - momentum
    v2 = np.add(beta*v2,alpha*np.matmul(rho_p,ReLU(z).T))
    c = np.subtract(c,v2)
    b2 = np.subtract(b2,alpha*np.mean(rho_p,axis=1,keepdims=True))
    
    v1 = np.add(beta*v1,alpha*np.matmul(np.multiply(delta,ReLU_p(z)),x))
    b1 = np.subtract(b1,alpha*np.mean(np.multiply(delta,ReLU_p(z)),axis=1,keepdims=True))
    w = np.subtract(w,v1)

# find accuracy on training set
p1 = 0
n1 = trn_y.shape[0]
for m1 in range(n1):
    z_pred1 = np.add(np.matmul(w,trn_x[[m1]].T),b1)
    u_pred1 = np.add(np.matmul(c,ReLU(z_pred1)),b2)
    s_pred1 = softmax(u_pred1)
    y_pred1 = np.where(s_pred1==np.amax(s_pred1))[0]
    if y_pred1 == trn_y[m1]:
        p1 += 1
print('Training Accuracy is :',p1/n1)

# find accuracy on testing set 
p2 = 0
n2 = tst_y.shape[0]
for m2 in range(n2):
    z_pred2 = np.add(np.matmul(w,tst_x[[m2]].T),b1)
    u_pred2 = np.add(np.matmul(c,ReLU(z_pred2)),b2)
    s_pred2 = softmax(u_pred2)
    y_pred2 = np.where(s_pred2==np.amax(s_pred2))[0]
    if y_pred2 == tst_y[m2]:
        p2 += 1
print('Testing Accuracy is :',p2/n2)

