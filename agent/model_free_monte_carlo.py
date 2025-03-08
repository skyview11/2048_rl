from torch import nn
import torch
import random

class QFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 1
    


class Trainer:
    def __init__(self):
        self.Q = QFunction()
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.steps_done = 0
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            torch.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold: # max q value action
            with torch.no_grad():
                return self.Q(state).max(1).indices.view(1, 1)
        else: # exploration
            return self.env.get_random_action()
            
        