import json
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt
import random

def load_colormap():
    with open("colormap.json") as f:
        colormap = json.load(f)
    return colormap

class MainBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.colormap = load_colormap()
        self.boardstate = [-1] * 16
        self.freeblocks = list(range(16))
        self.initUI()
        
    def initUI(self):
        # 처음에 블럭 2개로 시작
        self.update_new_block()
        self.update_new_block()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        ## wall 그리기 단계
        wall = self.colormap["board"]["wall"]
        painter.setBrush(QColor(wall[0], wall[1], wall[2]))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, 500, 500)
        ## block 그리기 단계
        for id, blockstate in enumerate(self.boardstate):
            row = id // 4
            col = id % 4
            if blockstate == -1: ## block이 비어있을 때
                bg_color = self.colormap["board"]["background"]
                painter.setBrush(QColor(bg_color[0], bg_color[1], bg_color[2]))
                painter.setPen(Qt.NoPen)
                painter.drawRect(20+col*120, 20+row*120, 100, 100)
            else: ## block에 어떤 숫자가 할당되었을때
                # block 색칠
                bg_color = self.colormap["block"][str(blockstate)]
                painter.setBrush(QColor(bg_color[0], bg_color[1], bg_color[2]))
                painter.setPen(Qt.NoPen)
                painter.drawRect(20+col*120, 20+row*120, 100, 100)
                # block에 글씨 넣기
                font = QFont("Arial", 40)
                pen = QPen(QColor(100, 100, 100))
                painter.setPen(pen)
                painter.setFont(font)
                painter.drawText(50+col*120, 85+row*120, str(blockstate))
        
    def update_new_block(self, new_id=None):
        if new_id is None:
            new_id = random.choice(self.freeblocks)
        assert self.boardstate[new_id] == -1
        self.boardstate[new_id] = 4 if random.random() < 0.25 else 2
        self.freeblocks.remove(new_id)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainBoard()
    window.show()
    sys.exit(app.exec())