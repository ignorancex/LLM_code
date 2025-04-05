import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.linalg as lin
import re
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
from scipy import signal
fsamp = 250
tsample = 1 / fsamp
f_low = 50
f_high = 1
order = 2
channel_vector = [1,2, 3, 4, 5]
n_ch = len(channel_vector)
# df = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_kinect__angles_.csv")
# df = df[channel_vector].dropna(axis=0)
#
# processed_signal = df.copy()

# b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))
#
# b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal.ix[:, i]))
#
# Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
# b3, a3 = butter(order, Wn, btype='stop')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal.ix[:, i]))

start = 850
end = 900

processed_signal = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_kinect__angles_.csv").dropna()


def nomalize_signal(input_signal):
    mean = np.mean(input_signal, axis=0)
    input_signal -= mean
    return input_signal / np.std(input_signal, axis=0)

#
# x = np.arange(start, end, 1)

# plt.plot(x, processed_signal.ix[:,1][start:end])
# plt.show()

processed_signal = nomalize_signal(processed_signal)



# plt.plot(x, processed_signal.ix[:,1][start:end])
# plt.show()

pattern=np.array(processed_signal.ix[:, 1][start :end])
data=np.array(processed_signal.ix[:, 1][750 :1300])

def create_mats(dat):
    '''
    create
        A - an initial transition matrix
        pA - pseudocounts for A
        w - emission distribution regression weights
        K - number of hidden states
    '''
    step=5 #adjust this to change the granularity of the pattern
    eps=.1
    dat=dat[::step]
    K=len(dat)+1
    A=np.zeros( (K,K) )
    A[0,1]=1.
    pA=np.zeros( (K,K) )
    pA[0,1]=1.
    for i in xrange(1,K-1):
        A[i,i]=(step-1.+eps)/(step+2*eps)
        A[i,i+1]=(1.+eps)/(step+2*eps)
        pA[i,i]=1.
        pA[i,i+1]=1.
    A[-1,-1]=(step-1.+eps)/(step+2*eps)
    A[-1,1]=(1.+eps)/(step+2*eps)
    pA[-1,-1]=1.
    pA[-1,1]=1.

    w=np.ones( (K,2) , dtype=np.float)
    w[0,1]=dat[0]
    w[1:-1,1]=(dat[:-1]-dat[1:])/step
    w[-1,1]=(dat[0]-dat[-1])/step

    return A,pA,w,K

#initialize stuff
A,pA,w,K=create_mats(pattern)

eta=5. #precision parameter for the autoregressive portion of the model
lam=.1 #precision parameter for the weights prior

N=1 #number of sequences
M=2 #number of dimensions - the second variable is for the bias term
T=len(data) #length of sequences

x=np.ones( (T+1,M) ) # sequence data (just one sequence)
x[0,1]=1
x[1:,0]=data

#emissions
e=np.zeros( (T,K) )
#residuals
v=np.zeros( (T,K) )

#store the forward and backward recurrences
f=np.zeros( (T+1,K) )
fls=np.zeros( (T+1) )
f[0,0]=1
b=np.zeros( (T+1,K) )
bls=np.zeros( (T+1) )
b[-1,1:]=1./(K-1)

#hidden states
z=np.zeros( (T+1),dtype=np.int )

#expected hidden states
ex_k=np.zeros( (T,K) )

# expected pairs of hidden states
ex_kk=np.zeros( (K,K) )
nkk=np.zeros( (K,K) )

def fwd(xn):
    global f,e
    for t in xrange(T):
        f[t+1,:]=np.dot(f[t,:],A)*e[t,:]
        sm=np.sum(f[t+1,:])
        fls[t+1]=fls[t]+np.log(sm)
        f[t+1,:]/=sm
        assert f[t+1,0]==0

def bck(xn):
    global b,e
    for t in xrange(T-1,-1,-1):
        b[t,:]=np.dot(A,b[t+1,:]*e[t,:])
        sm=np.sum(b[t,:])
        bls[t]=bls[t+1]+np.log(sm)
        b[t,:]/=sm

def em_step(xn):
    global A,w,eta
    global f,b,e,v
    global ex_k,ex_kk,nkk

    x=xn[:-1] #current data vectors
    y=xn[1:,:1] #next data vectors predicted from current
    #compute residuals
    v=np.dot(x,w.T) # (N,K) <- (N,1) (N,K)
    v-=y
    e=np.exp(-eta/2*v**2,e)

    fwd(xn)
    bck(xn)

    # compute expected hidden states
    for t in xrange(len(e)):
        ex_k[t,:]=f[t+1,:]*b[t+1,:]
        ex_k[t,:]/=np.sum(ex_k[t,:])

    # compute expected pairs of hidden states
    for t in xrange(len(f)-1):
        ex_kk=A*f[t,:][:,np.newaxis]*e[t,:]*b[t+1,:]
        ex_kk/=np.sum(ex_kk)
        nkk+=ex_kk

    # max w/ respect to transition probabilities
    A=pA+nkk
    A/=np.sum(A,1)[:,np.newaxis]

    # solve the weighted regression problem for emissions weights
    #  x and y are from above
    for k in xrange(K):
        ex=ex_k[:,k][:,np.newaxis]
        dx=np.dot(x.T,ex*x)
        dy=np.dot(x.T,ex*y)
        dy.shape=(2)
        w[k,:]=lin.solve(dx+lam*np.eye(x.shape[1]), dy)

    #return the probability of the sequence (computed by the forward algorithm)
    return fls[-1]

if __name__=='__main__':
    #run the em algorithm
    for i in xrange(5):
        print em_step(x)

    #get rough boundaries by taking the maximum expected hidden state for each position
    r=np.arange(len(ex_k))[np.argmax(ex_k,1)<2]
    f = np.diff(np.diff(r))
    for i in range(0,len(f)):
        if(f[i]<=0):
            r[i] = 0
    #plot
    plt.plot(range(T),x[1:,0])

    yr=[np.min(x[:,0]),np.max(x[:,0])]
    previous_value= r[0]
    plt.plot([r[0], r[0]], yr, '-r')

    for i in r:
            plt.plot([i,i],yr,'-r')
            previous_value = i

    plt.show()

# def create_mats(dat):
#     '''
#     create
#         A - an initial transition matrix
#         pA - pseudocounts for A
#         w - emission distribution regression weights
#         K - number of hidden states
#     '''
#     step=5 #adjust this to change the granularity of the pattern
#     eps=.1
#     dat=dat[::step]
#     K=len(dat)+1
#     A=np.zeros((K,K))
#     A[0,1]=1.
#     pA=np.zeros((K,K))
#     pA[0,1]=1.
#     for i in xrange(1,K-1):
#         A[i,i]=(step-1.+eps)/(step+2*eps)
#         A[i,i+1]=(1.+eps)/(step+2*eps)
#         pA[i,i]=1.
#         pA[i,i+1]=1.
#     A[-1,-1]=(step-1.+eps)/(step+2*eps)
#     A[-1,1]=(1.+eps)/(step+2*eps)
#     pA[-1,-1]=1.
#     pA[-1,1]=1.
#
#     w=np.ones( (K,2) , dtype=np.float)
#     w[0,1]=dat[0]
#     w[1:-1,1]=(dat[:-1]-dat[1:])/step
#     w[-1,1]=(dat[0]-dat[-1])/step
#
#     return A,pA,w,K
#
# #initialize stuff
# A,pA,w,K=create_mats(pattern)
#
# eta=10. #precision parameter for the autoregressive portion of the model
# lam=.1 #precision parameter for the weights prior
#
# N=1 #number of sequences
# M=2 #number of dimensions - the second variable is for the bias term
# T=len(data) #length of sequences
#
# x=np.ones( (T+1,M) ) # sequence data (just one sequence)
# x[0,1]=1
# x[1:,0]=data
#
# #emissions
# e=np.zeros( (T,K) )
# #residuals
# v=np.zeros( (T,K) )
#
# #store the forward and backward recurrences
# f=np.zeros( (T+1,K) )
# fls=np.zeros( (T+1) )
# f[0,0]=1
# b=np.zeros( (T+1,K) )
# bls=np.zeros( (T+1) )
# b[-1,1:]=1./(K-1)
#
# #hidden states
# z=np.zeros( (T+1),dtype=np.int )
#
# #expected hidden states
# ex_k=np.zeros( (T,K) )
#
# # expected pairs of hidden states
# ex_kk=np.zeros( (K,K) )
# nkk=np.zeros( (K,K) )
#
# def fwd(xn):
#     global f,e
#     for t in xrange(T):
#         f[t+1,:]=np.dot(f[t,:],A)*e[t,:]
#         sm=np.sum(f[t+1,:])
#         sm=sm+0.000000000001
#         fls[t+1]=fls[t]+np.log(sm)
#         f[t+1,:]/=sm
#         assert f[t+1,0]==0
#
# def bck(xn):
#     global b,e
#     for t in xrange(T-1,-1,-1):
#         b[t,:]=np.dot(A,b[t+1,:]*e[t,:])
#         sm=np.sum(b[t,:])
#         sm = sm + 0.000000000001
#         bls[t]=bls[t+1]+np.log(sm)
#         b[t,:]/=sm
#
# def em_step(xn):
#     global A,w,eta
#     global f,b,e,v
#     global ex_k,ex_kk,nkk
#
#     x=xn[:-1] #current data vectors
#     y=xn[1:,:1] #next data vectors predicted from current
#     #compute residuals
#     v=np.dot(x,w.T) # (N,K) <- (N,1) (N,K)
#     v-=y
#     e=np.exp(-eta/2*v**2,e)
#
#     fwd(xn)
#     bck(xn)
#
#     # compute expected hidden states
#     for t in xrange(len(e)):
#         try:
#             ex_k[t,:]=f[t+1,:]*b[t+1,:]
#             ex_k[t,:]/=np.sum(ex_k[t,:])+ 0.0000001
#         except :
#             print "exception  first "+ str(t)
#
#     # compute expected pairs of hidden states
#     for t in xrange(len(f)-1):
#         try:
#             ex_kk=A*f[t,:][:,np.newaxis]*e[t,:]*b[t+1,:]
#             ex_kk/=np.sum(ex_kk)+ 0.0000001
#             nkk+=ex_kk
#         except:
#             print "exception  second " + str(t)
#
#     # max w/ respect to transition probabilities
#     A=pA+nkk
#     A/=np.sum(A,1)[:,np.newaxis]
#
#     # solve the weighted regression problem for emissions weights
#     #  x and y are from above
#     for k in xrange(K):
#         ex=ex_k[:,k][:,np.newaxis]
#         dx=np.dot(x.T,ex*x)
#         dy=np.dot(x.T,ex*y)
#         dy.shape=(2)
#         w[k,:]=lin.solve(dx+lam*np.eye(x.shape[1]), dy)
#
#     #return the probability of the sequence (computed by the forward algorithm)
#     return fls[-1]
#
# if __name__=='__main__':
#     #run the em algorithm
#     for i in xrange(20):
#         print em_step(x)
#
#     #get rough boundaries by taking the maximum expected hidden state for each position
#     r=np.arange(len(ex_k))[np.argmax(ex_k,1)<3]
#
#     # plot
#     plt.plot(range(T),x[1:,0])
#
#     yr=[np.min(x[:,0]),np.max(x[:,0])]
#     # for i in range(0,10):
#     #     plt.plot([r[i],r[i]],yr,'-r')
#     for i in r:
#         plt.plot(i,i,yr,'-r')
#     plt.show()