import json
import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QEventLoop, pyqtSignal
import random
from debugtools import dprint
from collections import defaultdict

def load_colormap():
    with open("colormap.json") as f:
        colormap = json.load(f)
    return colormap


class BlockUnit(QLabel):
    animate_time = 100
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
    
    def moveto(self, new_id, loop):
        row = new_id // 4
        col = new_id % 4
        prev_row = self.row
        prev_col = self.col
        self.row = row
        self.col = col
        dprint(f"({prev_row}, {prev_col}) -> ({row}, {col})")
        ## animation
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(self.animate_time)  # 2초 동안 애니메이션
        self.anim.setStartValue(QRect(20+prev_col*120, 20+prev_row*120, 100, 100))
        self.anim.setEndValue(QRect(20+col*120, 20+row*120, 100, 100))
        self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.anim.finished.connect(loop.quit)
        self.anim.start()
        # self.setGeometry(20+col*120, 20+row*120, self.unitsize[0], self.unitsize[1])
    
    def update(self, blockstate, color):
        self.blockstate = blockstate
        self.setText(str(blockstate))
        self.setStyleSheet(f"background-color: rgb{tuple(color)}; color: rgb(100, 100, 100); font-size: 40px; padding: 10px;")

    
def skip():
    return
            
class MainBoard(QWidget):
    scoreChangeSig = pyqtSignal(int)
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
        self.events = []
        self.event_handling = False
        self.closed = False
        self.__score = 0
        self.__gameoverflag = False
        self.setParent(parent)
        self.initUI()
        
    def initUI(self):
        # 처음에 블럭 2개로 시작
        self.update_new_block()
        self.update_new_block()
        self.resize(500, 500)
    def closeEvent(self, event):
        self.closed = True
        
        event.accept()
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
        dprint(f"new block created in ({new_id // 4}, {new_id % 4}), value: {self.boardstate[new_id]}")
        
    def moveUpEvent(self):
        self.move_log = []
        integrated_blocks = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate.copy()
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
            if i-4 >= 0 and (i-4 not in integrated_blocks) and self.boardstate[i] == self.boardstate[i-4]:
                self.boardstate[i-4] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i -= 4
                gone = True
                integrated_blocks.append(i)
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
        # self.update()  # 다시 그리기 요청
        
    def moveDownEvent(self):
        self.move_log = []
        integrated_blocks = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate.copy()
        
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
            if i + 4 < 16 and (i + 4 not in integrated_blocks) and self.boardstate[i] == self.boardstate[i+4]:
                self.boardstate[i+4] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i += 4
                gone = True
                integrated_blocks.append(i)
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
            
        # self.update()  # 다시 그리기 요청
        
    def moveLeftEvent(self):
        self.move_log = []
        integrated_blocks = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate.copy()
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
            if i%4 != 0 and (i - 1 not in integrated_blocks) and self.boardstate[i] == self.boardstate[i-1]:
                dprint(f"id {id} will be integrated")
                self.boardstate[i-1] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i -= 1
                gone = True
                integrated_blocks.append(i)
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
        # self.update()  # 다시 그리기 요청
        
    def moveRightEvent(self):
        self.move_log = []
        integrated_blocks = []
        self.action_success = False
        prev_boardstate_buffer = self.boardstate.copy()
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
            if i%4 != 3 and (i + 1 not in integrated_blocks) and self.boardstate[i] == self.boardstate[i+1]:
                self.boardstate[i+1] *= 2
                self.boardstate[i] = -1
                self.freeblocks.append(i)
                self.action_success = True
                i += 1
                gone = True
                integrated_blocks.append(i)
            if i != id:
                self.move_log.append((id, i, gone))
        if self.action_success:
            self.prev_boardstate = prev_boardstate_buffer
        # self.update()  # 다시 그리기 요청
    
    def keyPressEvent(self, event):
        self.events.append(event.key())
        while not self.event_handling and len(self.events):
            self.keyPressEventHandler()
        else:
            pass
            # print("delayed")

    def keyPressEventHandler(self):
        self.event_handling = True
        event = self.events.pop()
        if event == Qt.Key_W:
            self.moveUpEvent()
        elif event == Qt.Key_A:
            self.moveLeftEvent()
        elif event == Qt.Key_S:
            self.moveDownEvent()
        elif event == Qt.Key_D:
            self.moveRightEvent()
        else:
            return
        loop = QEventLoop()
        for start, end, gone in self.move_log:
            # dprint(start, end)
            ## animation
            self.blocks[start].moveto(end, loop)
        # loop.exec_()
        if len(self.move_log):
            loop.exec_()
        added_score = 0
        for start, end, gone in self.move_log:
            if gone:
                assert end in self.blocks
                # self.blocks[start].deleteLater()
                label2del = self.blocks.pop(start, None)
                added_score += self.blocks[end].blockstate*2
                label2del.hide()
                label2del.deleteLater()
                self.blocks[end].update(self.blocks[end].blockstate*2, self.colormap["block"][str(self.blocks[end].blockstate*2)])
            else:
                self.blocks[end] = self.blocks.pop(start)
        dprint(f"Score: {self.__score}")
        self.__updateScore(added_score)
        if self.action_success:
            self.update_new_block()
        
        ## game over check (simulate move)
        action_success_buffer = self.action_success
        boardstate_buffer = self.boardstate.copy()
        prev_boardstate_buffer = self.prev_boardstate.copy()
        free_block_buffer = self.freeblocks.copy()
        # movelog_buffer = self.move_log.copy()
        movable = False
        for simulate in [self.moveUpEvent, self.moveDownEvent, self.moveLeftEvent, self.moveRightEvent]:
            simulate()
            simulate_success = self.action_success
            ## return to original state
            self.action_success = action_success_buffer
            self.boardstate = boardstate_buffer.copy()
            self.prev_boardstate = prev_boardstate_buffer.copy()
            self.freeblocks = free_block_buffer.copy()
            # self.move_log = movelog_buffer
            if simulate_success:
                movable = True
                break
        if not movable:
            self.__gameoverflag = True
        self.event_handling = False
       
    def getScore(self):
        return self.__score 

    def __updateScore(self, num):
        self.__score += num
        self.scoreChangeSig.emit(0)
    
    def isGameOver(self):
        return self.__gameoverflag
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainBoard(None)
    window.show()
    sys.exit(app.exec())