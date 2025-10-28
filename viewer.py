import sys
from PyQt5.QtWidgets import QApplication

from app import GameApp

from agent.randomAgent import RandomAgent

from train import encode_state

from PyQt5.QtCore import Qt, pyqtSignal
EPISODES = 5000
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

ACTION_SPACE = [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]

if __name__ == "__main__":
    from board import MainBoard
    import sys
    from PyQt5.QtWidgets import QApplication, QWidget

    app = QApplication(sys.argv)
    env = MainBoard(None)
    import torch
    env.show()
    
    
    from train import DQNAgent
    
    agent = DQNAgent(state_dim=16, action_dim=4)
    agent.epsilon = 0
    state_dict = torch.load("exp/zero_gamma3/epoch_300.pth")
    agent.model.load_state_dict(state_dict)
    while True:
        ## get state
        state = encode_state(torch.tensor(env.boardstate))
        with torch.no_grad():
            q_values = agent.model(state)
            print(q_values)
            apply = input()
            print("act")
            action = agent.act(state)
            action_qt = ACTION_SPACE[action]
            next_state, reward, done = env.step(action_qt)
                
    sys.exit(app.exec_())