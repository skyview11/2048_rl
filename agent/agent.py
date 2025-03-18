from abc import ABC, abstractclassmethod
from PyQt5.QtCore import QObject
class Agent(QObject):
    def __init__(self):
        super().__init__()
    def policy(self, gamestate):
        pass
    def optimize(self, gamestate):
        pass
    def close(self):
        pass