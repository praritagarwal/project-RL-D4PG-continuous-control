#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class actor(nn.Module):
    def __init__(self, n_states = 33, n_actions = 4, n_hidden = 100, seed = 0):
        super().__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        
        torch.manual_seed(seed)
        self.l1 = nn.Linear(in_features = self.n_states, out_features = self.n_hidden)
        self.l2 = nn.Linear(in_features = self.n_hidden, out_features = self.n_hidden//2)
        self.l3 = nn.Linear(in_features = self.n_hidden//2, out_features = self.n_hidden//4)
        self.l4 = nn.Linear(in_features = self.n_hidden//4, out_features = self.n_hidden//8)
        self.l5 = nn.Linear(in_features = self.n_hidden//8, out_features = self.n_actions)
        
    def forward(self, state):
        x = F.selu(self.l1(state))
        x = F.selu(self.l2(x))
        x = F.selu(self.l3(x))
        x = F.selu(self.l4(x))
        x = torch.tanh(self.l5(x))
        return x


# In[ ]:


class critic(nn.Module):
    def __init__(self, n_states = 33, n_actions= 4, n_hidden = 100, seed = 0):
        super().__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        
        torch.manual_seed(seed)
        self.l1 = nn.Linear(in_features = self.n_states + self.n_actions, out_features = self.n_hidden)
        self.l2 = nn.Linear(in_features = self.n_hidden, out_features = self.n_hidden//2)
        self.l3 = nn.Linear(in_features = self.n_hidden//2, out_features = self.n_hidden//4)
        self.l4 = nn.Linear(in_features = self.n_hidden//4, out_features = self.n_hidden//8)
        self.l5 = nn.Linear(in_features = self.n_hidden//8, out_features = 1)
        
    def forward(self, states, actions):
        critic_input = torch.cat((states, actions), dim = 1)
        x = F.selu(self.l1(critic_input))
        x = F.selu(self.l2(x))
        x = F.selu(self.l3(x))
        x = F.selu(self.l4(x))
        x = self.l5(x)
        return x

