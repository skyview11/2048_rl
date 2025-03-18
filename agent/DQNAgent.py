from torch import nn
import torch.nn.functional as F
import torch
import random

from .agent import Agent
from gameState import GameState

from PyQt5.QtCore import Qt, pyqtSignal
import math

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
BATCH_SIZE = 128
GAMMA = 0.5
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 5000
TAU = 0.005
LR = 1e-4

class QFunction(nn.Module):
    def __init__(self, state_dim=16, action_dim=4):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
    
    def forward(self, x): 
        # input: state(board) info
        # output: reward for each action
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        y = self.layer3(x)
        return y

class DataStack:
    """
    n th index boardstate, score, gameoverflag 
    -> n th index action 
    -> n+1 th index boardstate, score, gameoverflag 
    """
    def __init__(self, n_stack=100, device='cpu'):
        super().__init__()
        self.device = device
        self.boardstate = torch.zeros((n_stack+1, 16), dtype=torch.float, device=device) - 1
        self.score = torch.zeros((n_stack+1, 1), dtype=torch.float, device=device)
        self.selected_action = torch.zeros((n_stack, 1), dtype=torch.int, device=device)
        self.gameoverflag = torch.zeros((n_stack+1, 1), dtype=torch.bool, device=device)
        self.n_merged_block = torch.zeros((n_stack, 1), dtype=torch.float, device=device)
        # self.reward = torch.zeros((n_stack, 1), dtype=torch.float, device=device)
    def update(self, new_boardstate, new_action, new_score, new_gameoverflag, new_n_merged_block):
        new_boardstate = torch.tensor(new_boardstate, dtype=torch.float, device=self.device).unsqueeze(0)
        self.boardstate = torch.cat((self.boardstate, new_boardstate), dim=0)[1:]
        new_action = torch.tensor([new_action], dtype=torch.int, device=self.device).unsqueeze(0)
        self.selected_action = torch.cat((self.selected_action, new_action), dim=0)[1:]
        new_score = torch.tensor([new_score], dtype=torch.float, device=self.device).unsqueeze(0)
        self.score = torch.cat((self.score, new_score), dim=0)[1:]
        new_gameoverflag = torch.tensor([new_gameoverflag], dtype=torch.bool, device=self.device).unsqueeze(0)
        self.gameoverflag = torch.cat((self.gameoverflag, new_gameoverflag), dim=0)[1:]
        new_n_merged_block = torch.tensor([new_n_merged_block], dtype=torch.float, device=self.device).unsqueeze(0)
        self.n_merged_block = torch.cat((self.n_merged_block, new_n_merged_block), dim=0)[1:]
        
class DQNAgent(Agent):
    score_plot = pyqtSignal(int)
    loss_plot = pyqtSignal(float)
    eps_plot = pyqtSignal(float)
    def __init__(self, save_period=10000, update_iter=100):
        super().__init__()
        self.Q = QFunction()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.Q.to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr = LR)
        self.iteration = 0
        self.save_period = save_period
        self.update_period = update_iter
        self.data_stack = DataStack(n_stack=update_iter, device=self.device)
        ## plot info
        self.plot_manager = PltManager(self)
        
    def policy(self, gamestate: GameState, train=False):
        boardstate = torch.tensor(gamestate.boardstate, dtype=torch.float32).to(self.device)
        able_actions = gamestate.able_actions
        if train:
            action = self.select_action(boardstate)
            return [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D][action]
        else:
            rewards = self.Q(boardstate)
            sorted_index = torch.argsort(rewards, descending=True)
            for idx in sorted_index:
                target_action = [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D][idx]
                if target_action in able_actions:
                    return target_action

    def optimize(self, gamestate: GameState):
        ## add data for n iteration and update once
        
        ## initial boardstate
        if self.iteration == 0:
            self.data_stack.boardstate[-1] = torch.tensor(gamestate.prev_boardstate, dtype=torch.float, device=self.device)
        ## add data
        self.data_stack.update(new_boardstate=gamestate.boardstate, \
            new_action=gamestate.selected_action, \
            new_score=gamestate.score, \
            new_gameoverflag=gamestate.gameoverflag,\
            new_n_merged_block=gamestate.n_merged_blocks)
        # print(self.iteration, self.data_stack.boardstate)
        ## optimize every n iter or game is overed
        if ((self.iteration+1) % self.update_period == 0):
            encoded_boardstate = self.encode_boardstate(self.data_stack.boardstate[:-1])
            encoded_next_boardstate = self.encode_boardstate(self.data_stack.boardstate[1:])
            ## inferenced state action value
            q_values: torch.Tensor = self.Q(encoded_boardstate)
            state_action_value = q_values[torch.arange(self.update_period), self.data_stack.selected_action.squeeze()].unsqueeze(1)
            ## gt state action value
            with torch.no_grad():
                next_state_value = torch.sort(self.Q(encoded_next_boardstate), descending=True).values[:,0].unsqueeze(1)
            reward = self.reward_function()
            expected_value = (next_state_value * GAMMA) + reward
            criterion = nn.L1Loss()
            loss: torch.Tensor = criterion(state_action_value, expected_value)
            loss_float = float(torch.mean(loss))
            # print(expected_value.flatten(), state_action_value.flatten(), loss_float)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ## save model every 100k iter
        if (self.iteration+1) % self.save_period == 0:
            torch.save(self.Q.state_dict(), f"./ckpt/iter_{(self.iteration+1) // 1000}k.pth")

        ## update score graph every game ends
        if gamestate.gameoverflag:
            self.score_plot.emit(gamestate.score)
        ## update loss graph every 1k iter
        if (self.iteration+1)%1000 == 0:
            self.loss_plot.emit(loss_float)
            self.eps_plot.emit(self.eps)
            
        
        self.iteration += 1 # 무조건 맨 뒤에
        
            
    def select_action(self, state):
        sample = random.random()
        
        #eps_threshold: 작을수록 랜덤함.
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * (self.iteration // self.update_period) / EPS_DECAY)
        self.eps = eps_threshold
        if sample > eps_threshold:
            with torch.no_grad():
                rewards = self.Q(state)
                return torch.argsort(rewards, descending=True)[0]
        else:
            return random.randint(0, 3)
    
    def reward_function(self):
        ## weights
        death_weight = 100
        merge_weight = 10
        score_weight = 0
        reward = 0
        
        prev_boardstate = self.data_stack.boardstate[:-1]
        boardstate = self.data_stack.boardstate[1:]
        prev_score = self.data_stack.score[:-1]
        score = self.data_stack.score[1:]
        n_merged_block = self.data_stack.n_merged_block
        ## death penalty
        death_penalty = self.data_stack.gameoverflag[1:].to(torch.float) * -death_weight
        reward += death_penalty
        ## score
        added_score = score - prev_score
        reward += torch.log(torch.maximum(added_score, torch.tensor(1, device=self.device))) * score_weight
        ## merge bonus
        reward += n_merged_block * merge_weight
        ## covar
        
        return reward

    def encode_boardstate(self, boardstate):
        # [print(int(i), end=", ") for i in boardstate[0]]
        # print("")
        encoded_boardstate = boardstate + ((boardstate==-1)*2) ## -1 -> 1
        encoded_boardstate = torch.log2(encoded_boardstate)
        encoded_boardstate[encoded_boardstate != 0] = \
            encoded_boardstate[encoded_boardstate != 0] - (torch.min(encoded_boardstate[encoded_boardstate != 0])-1)
        # [print(int(i), end=", ") for i in encoded_boardstate[0]]
        # print("")
        encoded_boardstate = encoded_boardstate / torch.max(encoded_boardstate)
        # print("=======================================")
        return encoded_boardstate.to(self.device)
    def close(self):
        self.plot_manager.close()
        
        
from PyQt5.QtCore import QThread, QTimer
import time

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
class PltManager(QThread):
    def __init__(self, parent):
        super().__init__()
        
        self.agent = parent
        self.setParent(parent)

        self.show_result=True   
        
        ## plot info
        self.score_stack = []
        self.loss_stack = []
        self.eps_stack = []
        
        self.update_freq = 1000
        # self.iteration = 0
        
        self.agent.score_plot.connect(self.plot_score)
        self.agent.loss_plot.connect(self.plot_loss)
        self.agent.eps_plot.connect(self.plot_eps)
    def run(self):
        pass
        # if self.iteration % self.update_freq == 0:
        #     self.plot_loss(self.update_freq)
        #     self.plot_eps(self.update_freq)
        # time.sleep(50 / 1e6)
    def close(self):
        plt.close(1)
        plt.close(2)
        plt.close(3)
    def update_status(self, score, loss, eps):
        self.score_stack.append()
    def plot_score(self, score):
        # print("Score")
        self.score_stack.append(score)
        plt.figure(1)
        plt.clf()
        plt.title("Score Gragh (Train)")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(np.array(self.score_stack))
        plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
        if is_ipython:
            if not self.show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    def plot_loss(self, loss):
        # print("Loss")
        self.loss_stack.append(loss)
        plt.figure(2)
        plt.clf()
        plt.title("Loss Gragh (Train)")
        plt.xlabel(f"Iter ({self.update_freq//1000}k)")
        plt.ylabel("Loss")
        plt.plot(np.array(self.loss_stack))
        plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
        if is_ipython:
            if not self.show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def plot_eps(self, eps):
        # print("eps")
        self.eps_stack.append(eps)
        plt.figure(3)
        plt.clf()
        plt.title("Eps Gragh (Train)")
        plt.xlabel(f"Iter ({self.update_freq//1000}k)")
        plt.ylabel("Eps")
        plt.plot(np.array(self.eps_stack))
        plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
        if is_ipython:
            if not self.show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())