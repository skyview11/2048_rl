import random
from PyQt5.QtCore import Qt, pyqtSignal
from .agent import Agent
from gameState import GameState
class RandomAgent:
    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
        random.seed(seed)
    
    def policy(self, gamestate: GameState):
        return random.choice(gamestate.able_actions)