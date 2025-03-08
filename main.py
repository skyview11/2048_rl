import sys
from PyQt5.QtWidgets import QApplication

from app import GameApp

from agent.randomAgent import RandomAgent

if __name__ == "__main__":
    app = QApplication(sys.argv)
    agent = RandomAgent()
    ex = GameApp(is_manual=False)
    ex.
    sys.exit(app.exec_())