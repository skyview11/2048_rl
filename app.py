import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

import json


from board import MainBoard

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        # 메뉴바 만들기
        self.initMenu()

        # 레이아웃
        self.main_widget = MainBoard()
        self.setCentralWidget(self.main_widget)

        # show
        self.setWindowTitle("Test")
        self.setGeometry(300, 300, 800, 800)
        self.show()


    def initMenu(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        ## file menu
        filemenu = menubar.addMenu('&File')
        exit_action = QAction("Exit", self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(qApp.quit)
        
        filemenu.addAction(exit_action)
        
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
            
    def moveUpEvent(self):
        print("UP")
        success = False
        for id in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            if self.main_widget.boardstate[id] == -1:
                continue
            i = id
            while i-4 >= 0 and self.main_widget.boardstate[i-4] == -1:
                self.main_widget.boardstate[i-4] = self.main_widget.boardstate[i]
                self.main_widget.boardstate[i] = -1 
                self.main_widget.freeblocks.remove(i-4)
                self.main_widget.freeblocks.append(i)
                success = True
                i -= 4
            if i-4 >= 0 and self.main_widget.boardstate[i] == self.main_widget.boardstate[i-4]:
                self.main_widget.boardstate[i-4] *= 2
                self.main_widget.boardstate[i] = -1
                self.main_widget.freeblocks.append(i)
                success = True
        if success:
            self.main_widget.update_new_block()
        self.update()  # 다시 그리기 요청
        
    def moveDownEvent(self):
        print("Down")
        success = False
        for id in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
            if self.main_widget.boardstate[id] == -1:
                continue
            i = id
            while i+4 < 16 and self.main_widget.boardstate[i+4] == -1:
                self.main_widget.boardstate[i+4] = self.main_widget.boardstate[i]
                self.main_widget.boardstate[i] = -1 
                self.main_widget.freeblocks.remove(i+4)
                self.main_widget.freeblocks.append(i)
                success = True
                i += 4
            if i + 4 < 16 and self.main_widget.boardstate[i] == self.main_widget.boardstate[i+4]:
                self.main_widget.boardstate[i+4] *= 2
                self.main_widget.boardstate[i] = -1
                self.main_widget.freeblocks.append(i)
                success = True
        if success:
            self.main_widget.update_new_block()
        self.update()  # 다시 그리기 요청
        
    def moveLeftEvent(self):
        print("Left")
        success = False
        for id in [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]:
            if self.main_widget.boardstate[id] == -1:
                continue
            i = id
            while (i-1) % 4 != 3 and self.main_widget.boardstate[i-1] == -1:
                print(id, i)
                self.main_widget.boardstate[i-1] = self.main_widget.boardstate[i]
                self.main_widget.boardstate[i] = -1 
                self.main_widget.freeblocks.remove(i-1)
                self.main_widget.freeblocks.append(i)
                success = True
                i -= 1
            if i-1 >= 0 and self.main_widget.boardstate[i] == self.main_widget.boardstate[i-1]:
                self.main_widget.boardstate[i-1] *= 2
                self.main_widget.boardstate[i] = -1
                self.main_widget.freeblocks.append(i)
                success = True
        if success:
            self.main_widget.update_new_block()
        self.update()  # 다시 그리기 요청
    def moveRightEvent(self):
        print("Right")
        success = False
        for id in [2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12]:
            if self.main_widget.boardstate[id] == -1:
                continue
            i = id
            while (i+1) % 4 != 0 and self.main_widget.boardstate[i+1] == -1:
                print(id, i)
                self.main_widget.boardstate[i+1] = self.main_widget.boardstate[i]
                self.main_widget.boardstate[i] = -1 
                self.main_widget.freeblocks.remove(i+1)
                self.main_widget.freeblocks.append(i)
                success = True
                i += 1
            if i+1 < 16 and self.main_widget.boardstate[i] == self.main_widget.boardstate[i+1]:
                self.main_widget.boardstate[i+1] *= 2
                self.main_widget.boardstate[i] = -1
                self.main_widget.freeblocks.append(i)
                success = True
        if success:
            self.main_widget.update_new_block()
        self.update()  # 다시 그리기 요청
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())