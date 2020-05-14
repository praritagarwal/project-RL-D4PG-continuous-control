#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class actor(nn.Module):
    def __init__(self, n_states = 33, n_actions = 4, n_hidden = 50, seed = 0):
        super().__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        
        torch.manual_seed(seed)
        self.l1 = nn.Linear(in_features = self.n_states, out_features = self.n_hidden)
        self.l2 = nn.Linear(in_features = self.n_hidden, out_features = self.n_hidden//2)
        self.l3 = nn.Linear(in_features = self.n_hidden//2, out_features = self.n_hidden//4)
        self.l4 = nn.Linear(in_features = self.n_hidden//4, out_features = self.n_actions)
        
    def forward(self, state):
        x = F.selu(self.l1(state))
        x = F.selu(self.l2(x))
        x = F.selu(self.l3(x))
        x = torch.tanh(self.l4(x))
        return x


# In[ ]:


class critic(nn.Module):
    def __init__(self, n_states = 33, n_actions= 4, n_atoms = 51, n_hidden = 300, seed = 0, 
                 output = 'logprob'):
        # output: whether output should be softmax or log_softmax
        super().__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        
        if output == 'logprob':
            self.act = F.log_softmax
        elif output == 'prob':
            self.act = F.softmax
        else:
            print('Wrong value for parameter: output. Please choose from either prob or logprob')
        
        torch.manual_seed(seed)
        self.l1 = nn.Linear(in_features = self.n_states + self.n_actions, out_features = self.n_hidden)
        self.l2 = nn.Linear(in_features = self.n_hidden, out_features = (2*self.n_hidden)//3)
        self.l3 = nn.Linear(in_features = (2*self.n_hidden)//3, out_features = self.n_hidden//3)
        self.l4 = nn.Linear(in_features = self.n_hidden//3, out_features = n_atoms)
        
    def forward(self, states, actions):
        critic_input = torch.cat((states, actions), dim = 1)
        x = F.selu(self.l1(critic_input))
        x = F.selu(self.l2(x))
        x = F.selu(self.l3(x))
        x = self.act(self.l4(x), dim = 1) # outputs the log_prob for 
                                               # each 'atom' of the categorical distribution
        return x

