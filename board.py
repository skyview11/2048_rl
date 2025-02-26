import json
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtProperty
import random
from debugtools import dprint

from collections import defaultdict

def load_colormap():
    with open("colormap.json") as f:
        colormap = json.load(f)
    return colormap

class MainBoardAnimation:
    def __init__(self, board_widget, move_log):
        self.movelog = move_log
        self.animations = []
        for log in move_log:
            var_name = f"block_{log[0]}"
            self.animations.append(QPropertyAnimation(board_widget, var_name.encode()))
            

class BlockUnit(QLabel):
    def __init__(self, parent, id, blockstate, color, size=(100, 100)):
        super().__init__()
        self.unitsize = size
        self.setParent(parent)
        row = id // 4
        col = id % 4
        ## 기본 정보
        self.row = row
        self.col = col
        self.blockstate = blockstate
        self.setAlignment(Qt.AlignCenter)
        self.setGeometry(20+col*120, 20+row*120, 100, 100)
        self.setStyleSheet(f"background-color: rgb{tuple(color)}; color: rgb(100, 100, 100); font-size: 40px; padding: 10px;")
        if blockstate != -1:
            self.setText(str(blockstate))
    
    def moveto(self, new_id):
        row = new_id // 4
        col = new_id % 4
        self.row = row
        self.col = col
        self.setGeometry(20+col*120, 20+row*120, self.unitsize[0], self.unitsize[1])
        pass
    
    def update(self, blockstate, color):
        self.blockstate = blockstate
        self.setText(str(blockstate))
        self.setStyleSheet(f"background-color: rgb{tuple(color)}; color: rgb(100, 100, 100); font-size: 40px; padding: 10px;")
class MainBoard(QWidget):
    def __init__(self, parent):
        """보드 상에서 id 규칙 (16진수 기준)
           
           0 1 2 3
           4 5 6 7
           8 9 a b
           c d e f
           
        """
        super().__init__()
        self.colormap = load_colormap()
        self.boardstate = [-1] * 16
        self.prev_boardstate = None
        self.move_log = [] # [(1, 5, True), ...] 형태. 1 -> 5 로의 이동이 생겼으며, 해당 블럭은 합체되어 사라짐. 
        self.action_success = False
        self.freeblocks = list(range(16))
        self.blocks = defaultdict(BlockUnit)
        self.initUI()
     
    def initUI(self):
        # 처음에 블럭 2개로 시작
        self.update_new_block()
        self.update_new_block()
        self.resize(500, 500)

    def paintEvent(self, event):
        painter = QPainter(self)
        ## wall 그리기 단계
        wall = self.colormap["board"]["wall"]
        painter.setBrush(QColor(wall[0], wall[1], wall[2]))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, 500, 500)
        ## block 그리기 단계
        for id in range(16):
            row = id // 4
            col = id % 4
            bg_color = self.colormap["board"]["background"]
            painter.setBrush(QColor(bg_color[0], bg_color[1], bg_color[2]))
            painter.setPen(Qt.NoPen)
            painter.drawRect(20+col*120, 20+row*120, 100, 100)
        
    def update_new_block(self, new_id=None):
        if new_id is None:
            new_id = random.choice(self.freeblocks)
        assert self.boardstate[new_id] == -1
        self.boardstate[new_id] = 4 if random.random() < 0.25 else 2
        self.freeblocks.remove(new_id)
        self.blocks[new_id] = BlockUnit(self, new_id, self.boardstate[new_id], self.colormap["block"][str(self.boardstate[new_id])])
        self.blocks[new_id].show()
        dprint("new block created in " + str(new_id) + ", value: " + str(self.boardstate[new_id]))
        
    def moveUpEvent(self):
        self.move_log = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate
        for id in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            gone = False
            if self.boardstate[id] == -1:
                continue
            i = id
            while i-4 >= 0 and self.boardstate[i-4] == -1:
                self.boardstate[i-4] = self.boardstate[i]
                self.boardstate[i] = -1 
                self.freeblocks.remove(i-4)
                self.freeblocks.append(i)
                self.action_success = True
                i -= 4
            if i-4 >= 0 and self.boardstate[i] == self.boardstate[i-4]:
                self.boardstate[i-4] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i -= 4
                gone = True
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
        # self.update()  # 다시 그리기 요청
    def moveDownEvent(self):
        self.move_log = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate
        
        for id in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
            gone = False
            if self.boardstate[id] == -1:
                continue
            i = id
            while i+4 < 16 and self.boardstate[i+4] == -1:
                self.boardstate[i+4] = self.boardstate[i]
                self.boardstate[i] = -1 
                self.freeblocks.remove(i+4)
                self.freeblocks.append(i)
                self.action_success = True
                i += 4
            if i + 4 < 16 and self.boardstate[i] == self.boardstate[i+4]:
                self.boardstate[i+4] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i += 4
                gone = True
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
            
        # self.update()  # 다시 그리기 요청
        
    def moveLeftEvent(self):
        self.move_log = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate
        for id in [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]:
            gone = False
            if self.boardstate[id] == -1:
                continue
            i = id
            while (i-1) % 4 != 3 and self.boardstate[i-1] == -1:
                self.boardstate[i-1] = self.boardstate[i]
                self.boardstate[i] = -1 
                self.freeblocks.remove(i-1)
                self.freeblocks.append(i)
                self.action_success = True
                i -= 1
            if i-1 >= 0 and self.boardstate[i] == self.boardstate[i-1]:
                self.boardstate[i-1] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i -= 1
                gone = True
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
        # self.update()  # 다시 그리기 요청
        
    def moveRightEvent(self):
        self.move_log = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate
        for id in [2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12]:
            gone = False
            if self.boardstate[id] == -1:
                continue
            i = id
            while (i+1) % 4 != 0 and self.boardstate[i+1] == -1:
                self.boardstate[i+1] = self.boardstate[i]
                self.boardstate[i] = -1 
                self.freeblocks.remove(i+1)
                self.freeblocks.append(i)
                self.action_success = True
                i += 1
            if i+1 < 16 and self.boardstate[i] == self.boardstate[i+1]:
                self.boardstate[i+1] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i += 1
                gone = True
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
        # self.update()  # 다시 그리기 요청
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.moveUpEvent()
        elif event.key() == Qt.Key_A:
            self.moveLeftEvent()
        elif event.key() == Qt.Key_S:
            self.moveDownEvent()
        elif event.key() == Qt.Key_D:
            self.moveRightEvent()
        else:
            super().keyPressEvent(event)  # 기본 이벤트 처리
            return
        for start, end, gone in self.move_log:
            print(start, end)
            ## animation
            self.blocks[start].moveto(end)
            if gone:
                assert end in self.blocks
                self.blocks[start].deleteLater()
                self.blocks.pop(start, None)
                self.blocks[end].update(self.blocks[end].blockstate*2, self.colormap["block"][str(self.blocks[end].blockstate*2)])
            else:
                self.blocks[end] = self.blocks.pop(start)
        if self.action_success:
            self.update_new_block()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainBoard()
    window.show()
    sys.exit(app.exec())