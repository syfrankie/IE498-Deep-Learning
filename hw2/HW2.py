
# coding: utf-8


import numpy as np
import pandas as pd
import h5py
mnist = h5py.File('MNISTdata_1.hdf5','r')
list(mnist.keys())
trn_x,trn_y = np.array(mnist["x_train"][:]),np.array(mnist["y_train"][:])
tst_x,tst_y = np.array(mnist["x_test"][:]),np.array(mnist["y_test"][:])
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

#size of input layer
d0 = np.intc(np.sqrt(np.shape(trn_x)[1]))
#size of filter
ky = kx = 5
#number of channels
v = 32
#size of hidden layer
dhy = dhx = d0-ky+1
#size of output layer
c = 10

#input layer transformation
def input_trans(x,y):
    #x is the original input
    #y is size of filter
    #a is number of filter on each side of input
    #z is the transformed input
    a = np.shape(x)[0]-y+1
    z = np.zeros(shape=(y**2,a**2))
    for j in range(a):
        for i in range(a):
            z[:,a*j+i] = np.reshape(x[i:i+y,j:j+y].T,y**2)
    return z

#mean pooling function
def meanpool(x,m,n):
    a,b = np.shape(x)[0],np.shape(x)[1]
    y = np.zeros(shape=(a//m,b//n))
    for i in range(0,a,m):
        for j in range(0,b,n):
            y[i//m,j//n] = np.mean(x[i:i+m,j:j+n])
    return y
#backward mean pooling
def backpool(x,m,n):
    a,b = np.shape(x)[0],np.shape(x)[1]
    y = np.zeros(shape=(a*m,b*n))
    for i in range(0,a):
        for j in range(0,b):
            y[i*m:(i+1)*m,j*n:(j+1)*n] = np.mean(x[i,j])       
    return y
    

#initialize the parameters
np.random.seed(11)
k = np.random.normal(0,1/d0,size=(v,ky*kx))
w = np.random.normal(0,2/dhy,size=(c,v,dhy//2,dhx//2))
b = np.zeros(shape=(c,1))
alpha = 0.01

for l in range(20000):
    #forward
    #randomly select the index
    i = np.random.choice(trn_x.shape[0],size=1,replace=False)
    x0 = np.reshape(trn_x[[i]],(d0,d0))
    x = input_trans(x0,ky)
    z = np.matmul(k,x)
    #mean pooling
    zp = np.ones(shape=(v,dhy//2,dhx//2))
    for j in range(v):
        zp[j,:,:] = meanpool(np.reshape(z[j,:],(dhy,dhx)),2,2)
    h = ReLU(zp)
    u = np.add(np.reshape(np.einsum('ijkl,jkl->i',w,h),(c,1)),b)
    #rho = -np.log(softmax(u))
    
    #backward
    #define true y
    y = np.zeros(shape=(c,1))
    y[[trn_y[i]]] = 1
    # partial derivative of rho on u
    rho_p = -np.subtract(y,softmax(u))
    delta = np.reshape(np.einsum('im,ijkl->mjkl',rho_p,w),(v,dhy//2,dhx//2))
    #update parameters
    b = np.subtract(b,alpha*rho_p)
    w = np.subtract(w,alpha*np.einsum('lm,ijk->lijk',rho_p,h))
    #backpool
    dot = np.multiply(ReLU_p(zp),delta)
    kp = np.ones(shape=(v,dhy,dhx))
    for n in range(v):
        kp[n,:,:] = backpool(np.reshape(dot[n,:,:],(dhy//2,dhx//2)),2,2)
    kp = np.einsum('ijk->ij',np.reshape(kp,(v,dhy*dhx,1)))
    xb = input_trans(x0,dhy)
    k = np.subtract(k,alpha*np.matmul(kp,xb))

p1 = 0
n1 = trn_y.shape[0]
for m1 in range(n1):
    x01 = np.reshape(trn_x[[m1]],(d0,d0))
    x1 = input_trans(x01,ky)
    z_pred1 = np.matmul(k,x1)
    zp_pred1 = np.ones(shape=(v,dhy//2,dhx//2))
    for j in range(v):
        zp_pred1[j,:,:] = meanpool(np.reshape(z_pred1[j,:],(dhy,dhx)),2,2)
    h_pred1 = ReLU(zp_pred1)
    u_pred1 = np.add(np.reshape(np.einsum('ijkl,jkl->i',w,h_pred1),(c,1)),b)
    s_pred1 = softmax(u_pred1)
    y_pred1 = np.where(s_pred1==np.amax(s_pred1))[0]
    if y_pred1 == trn_y[m1]:
        p1 += 1
print('Training Accuracy is :',p1/n1)

p2 = 0
n2 = tst_y.shape[0]
for m2 in range(n2):
    x02 = np.reshape(tst_x[[m2]],(d0,d0))
    x2 = input_trans(x02,ky)
    z_pred2 = np.matmul(k,x2)   
    zp_pred2 = np.ones(shape=(v,dhy//2,dhx//2))
    for j in range(v):
        zp_pred2[j,:,:] = meanpool(np.reshape(z_pred2[j,:],(dhy,dhx)),2,2)
    h_pred2 = ReLU(zp_pred2)
    u_pred2 = np.add(np.reshape(np.einsum('ijkl,jkl->i',w,h_pred2),(c,1)),b)
    s_pred2 = softmax(u_pred2)
    y_pred2 = np.where(s_pred2==np.amax(s_pred2))[0]
    if y_pred2 == tst_y[m2]:
        p2 += 1
print('Testing Accuracy is :',p2/n2)



