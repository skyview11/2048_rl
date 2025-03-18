from torch import nn
from torch.nn import functional as F
from torch.optim import adam
class NNAgent(nn.Module):
    def __init__(self, state_dim=16, action_dim=4):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x))
        return x
        