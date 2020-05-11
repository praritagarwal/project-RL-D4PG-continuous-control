#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


from random import sample
from collections import deque
from Models import actor
from Models import critic


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


class Agent():
    def __init__(self, n_states = 33, n_actions = 4, actor_hidden = 100, 
                 critic_hidden = 100, seed = 0, roll_out = 5, replay_buffer_size = 1e6, 
                 replay_batch = 128, lr_actor = 5e-5,  lr_critic = 5e-5, epsilon = 0.3, 
                 tau = 1e-3,  gamma = 1, update_interval = 4, noise_fn = np.random.normal):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_hidden = actor_hidden # hidden nodes in the 1st layer of actor network
        self.critic_hidden = critic_hidden # hidden nodes in the 1st layer of critic network
        self.seed = seed
        self.roll_out = roll_out # roll out steps for n-step bootstrap; taken to be same as in D4PG paper
        self.replay_buffer = replay_buffer_size
        self.replay_batch = replay_batch # batch of memories to sample during training
        self.lr_actor = lr_actor # this was taken to same as the value in the D4PG paper for hard tasks
        self.lr_critic = lr_critic # taken from the D4PG paper
        self.epsilon = epsilon # to scale the noise before mixing with the actions; same as in D4PG paper
        self.tau = tau # for soft updates of the target networks
        self.gamma = gamma # do not decrease this below 1
        # note that we want the reacher to stay in goal position as long as possible
        # thus keeping gamma = 1 will ecourage the agent to increase its holding time
        self.update_every = update_interval # steps between successive updates
        self.noise = noise_fn # noise function; 
        # Note D4PG paper reported that 
        # using normal distribution instead of OU noise does not affect performance
        # will also experiment with OU noise if the need arises
        
        
        self.local_actor = actor(self.n_states, self.n_actions, self.actor_hidden, self.seed).to(device)
        self.local_critic = critic(self.n_states, self.n_actions, self.critic_hidden, self.seed).to(device)
        
        self.target_actor = actor(self.n_states, self.n_actions, self.actor_hidden, self.seed).to(device)
        self.target_critic = critic(self.n_states, self.n_actions, self.critic_hidden, self.seed).to(device)
        
        # initialize target_actor and target_critic weights to be 
        # the same as the corresponding local networks
        for target_c_params, local_c_params in zip(self.target_critic.parameters(), 
                                                   self.local_critic.parameters()):
            target_c_params.data.copy_(local_c_params.data)
        
        for target_a_params, local_a_params in zip(self.target_actor.parameters(), 
                                                   self.local_actor.parameters()):
            target_a_params.data.copy_(local_a_params.data)
            
        # optimizers for the local actor and local critic
        self.actor_optim = torch.optim.Adam(self.local_actor.parameters(), lr = self.lr_actor)
        self.critic_optim = torch.optim.Adam(self.local_critic.parameters(), lr = self.lr_critic)
        
        # loss function
        self.criterion = nn.MSELoss()
        
        # steps counter to keep track of steps passed between updates
        self.t_step = 0
        
        # replay memory 
        self.memory = ReplayBuffer(self.replay_buffer, self.n_states, 
                                   self.n_actions, self.roll_out)
    
    def act(self, states):
        # convert states to a torch tensor and move to the device
        # for the multiagent case we will get a batch of states 
        states = torch.from_numpy(states).float().to(device)
        self.local_actor.eval()
        with torch.no_grad():
            actions = self.local_actor(states).cpu().detach().numpy()
            noise = self.noise(size = actions.shape)
            actions = np.clip(actions + noise, -1, 1)
        self.local_actor.train()
        return actions
            
    def step(self, new_memories):
        # new memories is a batch of tuples
        # each tuple consists of (n-1)-steps of state, action, reward, done and the n-state
        # here n is the roll_out length
        self.memory.add(new_memories)
        
        # update the networks after every self.update_every steps
        # make sure to check that the replay_buffer has enough memories
        self.t_step = (self.t_step+1)%self.update_every
        if self.t_step == 0 and self.memory.__len__() > 2*self.replay_batch:
            self.learn()
    
    def learn(self):
        # sample a batch of memories from the replay buffer
        states_0, actions_0, rewards, states_fin = self.memory.sample(self.replay_batch)
        
        states_0 = torch.from_numpy(states_0).float().to(device)
        actions_0 = torch.from_numpy(actions_0).float().to(device)
        states_fin = torch.from_numpy(states_fin).float().to(device)
        
        # get an action for the n-th state from the target actor
        self.target_actor.eval()
        with torch.no_grad():
            actions_fin = self.target_actor(states_fin)
        self.target_actor.train()    
        
        # get the Q-value for the n-th state and action from the target critic
        self.target_critic.eval()
        with torch.no_grad():
            fin_Qs = self.target_critic(states_fin, actions_fin)
        self.target_critic.train()    
            
        
        # Compute the TD-target for the n-step bootstrap
        discounts = np.array([self.gamma**powr for powr in range(self.roll_out - 1 )])
        n_step_rewards = np.matmul(rewards, discounts.reshape(-1,1)) # sum of the discounted rewards collected during the roll_out
        n_step_rewards = torch.from_numpy(n_step_rewards).float().to(device)
        # fin_done = None # was the final state a terminal state?
        target_Q = n_step_rewards + (self.gamma**(self.roll_out -1))*fin_Qs
        
        # train the local critic
        self.critic_optim.zero_grad()
        # get a Q-value for the beginning state and action from the local critic
        local_Q = self.local_critic(states_0, actions_0)
        # compute the local critic's loss
        loss_c = self.criterion(local_Q, target_Q)
        loss_c.backward()
        self.critic_optim.step()
        
        # train the local actor
        self.actor_optim.zero_grad()
        # get the local_action for the initial state
        local_a = self.local_actor(states_0)
        # get the Q_value for the initial state and local_a
        # this gives the actor's loss
        loss_a = -torch.mean(self.local_critic(states_0, local_a)) 
        # this should be: - self.local_critic(initial_state, local_a)
        loss_a.backward()
        self.actor_optim.step()
        
        # apply soft updates to the target network
        self.update_target_networks()
      
    def update_target_networks(self):
        # update target actor
        for params_target, params_local in zip(self.target_actor.parameters(),
                                               self.local_actor.parameters()):
            updates = (1.0-self.tau)*params_target.data + self.tau*params_local.data 
            params_target.data.copy_(updates)
            
        # update target critic 
        for params_target, params_local in zip(self.target_critic.parameters(), 
                                               self.local_critic.parameters()):
            updates = (1.0-self.tau)*params_target.data + self.tau*params_local.data 
            params_target.data.copy_(updates)
        


# In[5]:


class ReplayBuffer():
    
    def __init__(self, buffer_size, n_states, n_actions, roll_out):
        self.memory = deque(maxlen = int(buffer_size))
        self.n_states = n_states
        self.n_actions = n_actions
        self.roll_out = roll_out
            
    def add(self, experience_windows):
        
        for window in experience_windows:
            self.memory.append(window)
    
    def sample(self, batch_size):
        batch = sample(self.memory, batch_size)
        
        # from the above batch obtain states at 0th step, action at 0th step, 
        # rewards for the n-1 subsequent steps i.e. rewards until and incuding the penultimate step
        # and the states at the last roll_out step
        
        states_0 = []
        actions_0 = []
        rewards = []
        states_fin = []
        for exp in batch:
            state_0 = exp[0][:self.n_states]
            action_0 = exp[0][self.n_states:self.n_states+self.n_actions]
            reward = [exp[ii][self.n_states+self.n_actions] for ii in range(self.roll_out-1)]
            state_fin = exp[self.roll_out-1][:self.n_states]
            states_0.append(state_0)
            actions_0.append(action_0)
            rewards.append(reward)
            states_fin.append(state_fin)
        
        states_0 = np.array(states_0)
        actions_0 = np.array(actions_0)
        rewards = np.array(rewards)
        states_fin = np.array(states_fin)
        
        return states_0, actions_0, rewards, states_fin
    
    def __len__(self):
        return len(self.memory)

