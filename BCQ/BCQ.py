#https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py

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
from torch.utils.data.dataset import IterableDataset

import gym 
from gym.wrappers import  Monitor
import pybullet_envs
#pip install git+https://github.com/takuseno/d4rl-pybullet
import d4rl_pybullet


#https://github.com/vwxyzjn/cleanrl/blob/c1fb51a7c32596ef3133af4617f92d2095051c66/cleanrl/offline_dqn_cql_atari_visual.py#L496    
class ExperienceReplayDataset(IterableDataset):
    def __init__(self):
        self.dataset_env = gym.make(offline_gym_id)
        self.dataset = self.dataset_env.get_dataset()
    def __iter__(self):
        while True:
            idx = np.random.choice(len(self.dataset['observations'])-1)
            yield self.dataset['observations'][idx],self.dataset['actions'][idx],self.dataset['rewards'][idx],self.dataset['observations'][idx+1],self.dataset['terminals'][idx]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,phi =0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.phi = phi
        
    def forward(self, state,action):
        a = F.relu(self.l1(torch.cat([state,action],1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a+action).clamp(-self.max_action,self.max_action)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        #Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


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

 
# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self,state_dim,action_dim,latent_dim,max_action,device):
        super(VAE,self).__init__()
        self.e1 = nn.Linear(state_dim+action_dim,750)
        self.e2 = nn.Linear(750,750)
        
        self.mean = nn.Linear(750,latent_dim)
        self.log_std = nn.Linear(750,latent_dim)
        
        self.d1 = nn.Linear(state_dim+latent_dim,750)
        self.d2 = nn.Linear(750,750)
        self.d3 = nn.Linear(750,action_dim)
        
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device
        
    def forward(self,state,action):
        z = F.relu(self.e1(torch.cat([state,action],1)))
        z = F.relu(self.e2(z))
        
        mean = self.mean(z)
        #log_std = self.log_std(z).clamp(-4,15)# Clamped for numerical stability 
        log_std = self.log_std(z)
        std = torch.exp(log_std)
        z = mean+std*torch.randn_like(std)
        
        u = self.decode(state,z)
        return u, mean, std 
        
    def decode(self,state,z=None):
        if z is None:
            #z = torch.randn((state.shape[0],self.latent_dim)).to(self.device).clamp(-0.5,0.5)
            z = torch.randn((state.shape[0],self.latent_dim)).to(self.device).clamp(-1.5,1.5)
        a = F.relu(self.d1(torch.cat([state,z],1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
       
        

class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device,
                 discount=0.99,
                 tau=0.005,
                 actor_lr = 1e-3,
                 critic_lr = 1e-3,
                 policy_noise =0.2,
                 noise_clip = 0.5,
                 lmbda = 0.75,
                 phi = 0.05,
                 policy_updatefreq=2):
        
        latent_dim = action_dim * 2
        
        self.actor = Actor(state_dim, action_dim, max_action,phi).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action,phi).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = critic_lr)

        self.vae = VAE(state_dim,action_dim,latent_dim,max_action,device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_updatefreq = policy_updatefreq
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat (100,1).to(self.device)
            action = self.actor(state,self.vae.decode(state))
            q1 = self.critic.Q1(state,action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()
    
    def train(self,data_loader,batch_size,current_step):
        batch_state,batch_action,batch_reward,batch_next_state,batch_not_done = next(data_loader)
        
        #torch conversion
        state = torch.FloatTensor(batch_state).to(self.device)
        action = torch.FloatTensor(batch_action).to(self.device)
        next_state = torch.FloatTensor(batch_next_state).to(self.device)
        reward = torch.FloatTensor(batch_reward).view(-1,1).to(self.device)
        not_done = torch.FloatTensor(1-batch_not_done).view(-1,1).to(self.device)
        # tensor conversion completed
        #print(f'reward:{reward.shape}')
        #print(f'state:{state.shape}')
        #print(f'not_done:{not_done.shape}')
        # Variational Auto-Encoder Training
        recon,mean,std = self.vae(state,action)
        recon_loss = F.mse_loss(recon,action)
        KL_loss = -0.5*(1+torch.log(std.pow(2))-  mean.pow(2)-std.pow(2)).mean()
        vae_loss = recon_loss+ 0.5*KL_loss
        writer.add_scalar("losses/VAEloss",vae_loss.item(),current_step)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state,10,0)
            
            next_action = self.actor_target(next_state, self.vae.decode(next_state))
            #print(f'next_action:{next_action.shape}')
            #Compute the target
            target_Q1,target_Q2 = self.critic_target(next_state,next_action)
            #print(f'target_Q1:{target_Q1.shape}')
            #print(f'target_Q2:{target_Q2.shape}')
            # Soft Clipped Double Q-learning 
            target_Q = self.lmbda * torch.min(target_Q1,target_Q2)+(1.-self.lmbda)*torch.max(target_Q1,target_Q2)
            #print(f'target_Q_min:{target_Q.shape}')
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size,-1).max(1)[0].reshape(-1,1)
            #print(f'target_Q_reshape:{target_Q.shape}')
            target_Q = reward + (not_done * self.discount*target_Q)
            #print(f'target_Q:{target_Q.shape}')
            
        #Get current Estimates
        current_Q1,current_Q2 = self.critic(state,action)
        #print(f'current_Q1:{current_Q1.shape}')
        #print(f'current_Q2:{current_Q2.shape}')
        #Compute Critic loass
        critic_loss = F.mse_loss(target_Q,current_Q1)+F.mse_loss(target_Q,current_Q2)
        #print(f'critic_loss:{critic_loss.shape}')
        writer.add_scalar("losses/criticloss",critic_loss.item(),current_step)
        #Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        
        if current_step % self.policy_updatefreq ==0:
            
            #compute actor loss(pertubation Model /Action training)
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)
            #update through DPG
            actor_loss = -self.critic.Q1(state,perturbed_actions).mean()
            
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


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name ,seed, eval_episodes=10):
    
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    return avg_reward

#selct the Gym enviornment
gym_id = "HopperBulletEnv-v0"
env = gym.make(gym_id)
offline_gym_id = 'hopper-bullet-medium-v0'
dataset_env = gym.make(offline_gym_id)
dataset_env.reset()

#set the seed
seed =100
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


actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
total_timesteps = 10000000
learning_starts = 0
batch_size = 100
policy_updatefreq = 1
eval_frequency = 1000

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_loader = iter(torch.utils.data.DataLoader(
    ExperienceReplayDataset(), batch_size=batch_size, num_workers=2))
#replay_buffer = ReplayBuffer()
BCQ_policy = BCQ(state_dim,action_dim,max_action,device,\
                   actor_lr=actor_learning_rate,critic_lr =critic_learning_rate )


suffix = f"BCQ_{offline_gym_id}_{int(time.time())}"
writer = SummaryWriter(f"runs/{suffix}")

# Time to train and collect Rewards

for global_step in range(total_timesteps):
    
    BCQ_policy.train(data_loader,batch_size,global_step)
        
    if global_step%eval_frequency ==0:
        episode_reward = eval_policy(BCQ_policy, gym_id, seed, eval_episodes=10)
        print(f"globalstep:{global_step} ; rewards :{episode_reward:.3f}")
        writer.add_scalar("rewards",episode_reward,global_step)
        
dataset_env.close()
writer.close()
        








