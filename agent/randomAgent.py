import random
from PyQt5.QtCore import Qt, pyqtSignal

class RandomAgent:
    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
        random.seed(seed)
    
    def policy(self):
        return random.choice([Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D])