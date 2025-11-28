from board import MainBoard
from PyQt5.QtWidgets import QApplication
import sys


import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import random
import time
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal


EPISODES = 50000
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995



class Memory:
    def __init__(self, mem_size, data_dim):
        self.data_dim = data_dim
        self.mem_size = mem_size
        self.memory = [torch.zeros(mem_size, dim) for dim in data_dim]
        self.n = 0
    def append(self, data):
        for i, d in enumerate(data):
            self.memory[i][self.n] = d
        self.n  = (self.n + 1)%self.mem_size
    
    def sample(self, k):
        indices = random.sample(range(k), k)
        return [self.memory[i][indices] for i in range(len(self.data_dim))]
    def __len__(self):
        return self.n+1
            

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
    def forward(self, x):
        return self.net(x)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path=None):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.memory = Memory(MEMORY_SIZE, (self.state_dim, 1, 1, self.state_dim, 1))
        self.epsilon = EPS_START
        
        self.model = DQN(state_dim, action_dim).to("cuda")
        self.target_model = DQN(state_dim, action_dim).to("cuda")
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        
        if model_path is not None:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.target_model.load_state_dict(state_dict)
    def act(self, state):
        
        q_values = self.model(state.float().to("cuda"))
        print(q_values.tolist(), end=" ")
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # batch = random.sample(self.memory, BATCH_SIZE)
        # import pdb;pdb.set_trace()
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = states.to("cuda")
        actions = actions.long().to("cuda")
        rewards = rewards.squeeze().to("cuda")
        next_states = next_states.to("cuda")
        dones = dones.squeeze().to("cuda")
        
        # Q(s, a)
        curr_Q = self.model(states).gather(1, actions).squeeze()
        
        # target Q -> GT
        next_Q = self.target_model(next_states).max(1)[0].detach()
        target_Q = rewards + GAMMA * next_Q * (1 - dones)
        
        # import pdb;pdb.set_trace()
        loss = self.loss_fn(curr_Q, target_Q)
        loss_avg = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_avg
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


if __name__ == '__main__':
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    events = [QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier) for key in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]]
    state_dim = 16
    action_dim = 4
    agent = DQNAgent(state_dim, action_dim, model_path="../2048_env/exp/run2_continue/epoch_latest.pth")


    app = QApplication(sys.argv)
    window = MainBoard(None)
    window.show()
    while True:
        action = agent.act(torch.tensor(window.boardstate))
        print(chr([Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D][action]), end="")
        input()
        window.keyPressEvent(events[action])
        
    sys.exit(app.exec())