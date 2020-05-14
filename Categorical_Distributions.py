#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
import matplotlib.pyplot as plt
import torch


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# function to compute the projected probablities
# This is an implementation of Algorithm 1 in the Distributional perpective paper: 
# arXiv:1707.06887 [cs.LG]
def projected_prob(vmin, vmax, N, reward, discount, target_prob):
    delta = (vmax - vmin)/(N-1)
    z = np.array([ vmin + i*delta for i in range(N)])
    Tz = np.clip(reward + discount*z, vmin, vmax)
    b = (Tz-vmin)/delta
    l = np.floor(b).astype(int)
    small_shift = 1e-5
    u = np.ceil(b+small_shift).astype(int)
    projected_probs = np.zeros(N)
    for ii, lu in enumerate(zip(l,u)):
        ll, uu = lu
        if ll in range(N):
            projected_probs[ll]+=target_prob[ii]*(uu-b[ii])
        if uu in range(N):
            projected_probs[uu]+=target_prob[ii]*(b[ii]-ll)
    return projected_probs     


# In[ ]:


# function to apply projected_prob to the rows of a batch
def projected_prob_batch(vmin, vmax, N, discount, rewards, target_probs):
    '''This is a much slower implementation. 
       Try projected_prob_batch2 instead.
       Average time taken by this function on a batch of 128 with N = 51: 0.112sec
       Average time taken by projected_prob_batch2 on the same computations: 0.0012sec'''
    rw_prb = np.concatenate((rewards, target_probs), axis = 1)
    fn = lambda x: projected_prob(vmin, vmax, N, x[0], discount, x[1:])
    return np.apply_along_axis(fn, 1, rw_prb)


# In[ ]:


def projected_prob_batch2(vmin, vmax, N, discount, rewards, target_probs):
    '''This is much faster implementation of projected_prob_batch
       Average time taken by this function on a batch of 128 with N = 51: 0.0012sec
       Average time taken by projected_prob_batch on the same computations: 0.112sec'''
    delta = (vmax - vmin)/(N-1)
    z = np.array([ vmin + i*delta for i in range(N)])
    # z = np.linspace(vmin, vmax, N)
    # I experimented with np.linspace to generate z but it seems to be much slower 
    Tz = np.clip(rewards + discount*z, vmin, vmax)
    b = (Tz-vmin)/delta
    l = np.floor(b).astype(int)
    small_shift = 1e-10
    u = np.ceil(b+small_shift).astype(int)
    batch_size = target_probs.shape[0]
    # since 'u' can contain a value N+1
    # we will therefore first create a zero matrix of shape (batch_size, N+1)
    # after appropriately updating it we will simply ignore its (N+1)th column 
    projected_probs = np.zeros((batch_size, N+1))
    for idx in range(N):
        ll, uu = l[:,idx], u[:, idx]
        projected_probs[np.arange(batch_size), ll]+=target_probs[:, idx]*(uu-b[:,idx])
        projected_probs[np.arange(batch_size), uu]+=target_probs[:, idx]*(b[:, idx]-ll)
    return projected_probs[:, :N]


# In[ ]:


def compare_computation_time(batch_size = 128, vmin = -10, vmax = 10, N = 51, discount = 1):
    # average the timing over 100 iterations
    time1 = []
    time2 = []
    for itr in range(100):
        rand = np.random.normal(size = (batch_size, N))
        exp =  np.exp(rand) 
        smi =   1/np.sum(exp, axis = 1)
        diag = np.zeros((batch_size,batch_size))
        diag[np.arange(batch_size), np.arange(batch_size)] = smi
        probs = np.matmul(diag, exp)
        if (np.abs(np.sum(probs, axis = 1)-1.0)>1e-10).any():
            print('problem in softmax function')
            return
        
        start1 = time.time()
        proj_probs1 = projected_prob_batch(vmin, vmax, N, discount, 
                                           rewards = np.ones((batch_size,1)), 
                                           target_probs = probs )
        end1 = time.time()
        time1.append(end1-start1)
        if (np.abs(np.sum(proj_probs1, axis = 1)-1.0)>1e-10).any():
            print('problem in projected_prob_batch')
            return
        
        start2 = time.time()
        proj_probs2 = projected_prob_batch2(vmin, vmax, N, rewards = np.ones((batch_size,1)), 
                                            discount = discount,  target_probs = probs )
        end2 = time.time()
        time2.append(end2-start2)
        if (np.abs(np.sum(proj_probs2, axis = 1)-1.0)>1e-10).any():
            print('problem in projected_prob_batch2')
            return
        if(np.abs(proj_probs1-proj_probs2)>1e-10).any():
            print('results of the two projections do not match ')
            return
        
    plt.figure()
    plt.plot(time1, label = 'projected_prob_batch')
    plt.plot(time2, label= 'projected_prob_batch2')
    plt.legend()
    plt.show()
    print('average time taken by projected_prob_batch: {}'.format(np.average(time1)))
    print('average time taken by projected_prob_batch2: {}'.format(np.average(time2)))


# In[ ]:


# same as projected_prob_batch2 but directly as torch tensors
def projected_prob_batch2_torch(vmin, vmax, N, discount, rewards, target_probs, batchsize):
    '''Same as projected_prob_batch2 but directly as torch tensors
        rewards and target_probs are assumed to be torch tensors'''
    delta = (vmax - vmin)/(N-1)
    z = torch.linspace(vmin, vmax, N).to(device) 
    Tz = torch.clamp(rewards + discount*z, vmin, vmax)
    b = (Tz-vmin)/delta
    l = torch.floor(b).type(torch.long)
    small_shift = 1e-5 # shifts smaller than this will not work for float32 tensors
    u = torch.ceil(b+small_shift).type(torch.long)
    # since 'u' can contain a value N+1
    # we will therefore first create a zero matrix of shape (batch_size, N+1)
    # after appropriately updating it we will simply ignore its (N+1)th column 
    projected_probs = torch.zeros((batchsize, N+1)).to(device)
    for idx in range(N):
        ll, uu = l[:,idx], u[:, idx]
        projected_probs[torch.arange(batchsize), ll]+=target_probs[:, idx]*(uu-b[:,idx])
        projected_probs[torch.arange(batchsize), uu]+=target_probs[:, idx]*(b[:, idx]-ll)
    return projected_probs[:, :N]

