import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import random

EPISODES = 500
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPS_START
        
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        
    def policy(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            q_values = self.model(torch.FloatTensor(state))
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Q(s, a)
        curr_Q = self.model(states).gather(1, actions).squeeze()
        
        # target Q -> GT
        next_Q = self.target_model(next_states).max(1)[0].detach()
        target_Q = rewards + GAMMA * next_Q * (1 - dones)

        loss = self.loss_fn(curr_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
 