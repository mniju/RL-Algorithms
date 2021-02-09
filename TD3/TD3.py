import torch
import torch.nn as nn
import numpy as np
import time
import os
import random
import torch.nn.functional as F

from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal

import gym 
from gym.wrappers import  Monitor
import pybullet_envs


class ReplayBuffer(object):
    
    def __init__(self,max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr =0
        
    def add(self,transition):
        if len(self.storage)==self.max_size: 
            self.storage[self.ptr] = transition  #lossy enque
            self.ptr = int((self.ptr+1) % self.max_size)
        else:
            self.storage.append(transition)
    def sample(self,batch_size):
        ind = np.random.randint(0,len(self.storage),size=batch_size)
        batch_states,batch_next_states,batch_actions,batch_rewards,batch_dones = [],[],[],[],[]
        for i in ind:
            state,next_state,action,reward,done = self.storage[i]
            batch_states.append(state)
            batch_next_states.append(next_state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_dones.append(done)
        return np.array(batch_states),np.array(batch_next_states),np.array(batch_actions),np.array(batch_rewards).reshape(-1,1),np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        #Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1,q2
    
    def Q1(self,state,action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
        
        

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, device,
                 discount=0.99,
                 tau=0.005,
                 actor_lr = 1e-3,
                 critic_lr = 1e-3,
                 policy_noise =0.2,
                 noise_clip = 0.5,
                 policy_updatefreq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_updatefreq = policy_updatefreq
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self,replay_buffer,batch_size,current_step):
        batch_state,batch_next_state,batch_action,batch_reward,batch_not_done = replay_buffer.sample(batch_size)
        
        #torch conversion

        state = torch.FloatTensor(batch_state).to(self.device)
        action = torch.FloatTensor(batch_action).to(self.device)
        next_state = torch.FloatTensor(batch_next_state).to(self.device)
        reward = torch.FloatTensor(batch_reward).to(self.device)
        not_done = torch.FloatTensor(batch_not_done).to(self.device)
        # tensor conversion completed
        with torch.no_grad():
            noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.policy_noise,self.policy_noise)
        
            next_action = (self.actor_target(next_state)+noise).clamp(-self.max_action,self.max_action)
            #Compute the target
            target_Q1,target_Q2 = self.critic_target(next_state,next_action)
            target_Q = torch.min(target_Q1,target_Q2)
            target_Q = reward + (not_done * self.discount*target_Q)
        #Get current Estimates
        current_Q1,current_Q2 = self.critic(state,action)
        #Compute Critic loass
        critic_loss = F.mse_loss(target_Q,current_Q1)+F.mse_loss(target_Q,current_Q2)
        
        writer.add_scalar("losses/criticloss",critic_loss.item(),current_step)
        #Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if current_step % self.policy_updatefreq ==0:
            
            #compute actor loss
            actor_loss = -self.critic.Q1(state,self.actor(state)).mean()
            writer.add_scalar("losses/actorloss",actor_loss.item(),current_step)
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#selct the Gym enviornment
gym_id = "HalfCheetahBulletEnv-v0"
env = gym.make(gym_id)

#set the seed
seed =43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

max_action = float(env.action_space.high[0])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  


actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
total_timesteps = 10000000
learning_starts = 25e3
batch_size = 32
policy_updatefreq = 2

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

replay_buffer = ReplayBuffer()
Td3_policy = TD3(state_dim,action_dim,max_action,device,\
                   actor_lr=actor_learning_rate,critic_lr =critic_learning_rate )

state = env.reset()
episode_reward =0

writer = SummaryWriter(f"runs/TD3_trial01")

# Time to train and collect Rewards

for global_step in range(total_timesteps):
    if global_step < learning_starts:
        #just explore here until we hit the learningh starts step
        action = env.action_space.sample()
    else:
        action = Td3_policy.select_action(np.array(state))
        
    next_state,reward,done,_ = env.step(action)
    episode_reward += reward
    replay_buffer.add((state,next_state,action,reward,1-done))
    
    if global_step>learning_starts:
        
        Td3_policy.train(replay_buffer,batch_size,global_step)

    state = next_state
    
    if done:
        print(f"globalstep:{global_step} ; rewards :{episode_reward}")
        writer.add_scalar("rewards",episode_reward,global_step)
        
        state,episode_reward  =env.reset(),0
        
env.close()
writer.close()







