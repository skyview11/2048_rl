import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtCore import Qt

from PyQt5.QtCore import Qt, QEvent


from debugtools import *

from gameState import GameState


from board import MainBoard, BlockUnit
from agenthandler import AgentHandler
class GameApp(QMainWindow):
    def __init__(self, is_manual=True):
        super().__init__()
        self.initUI()
        self.initState()
        self.initAgent(is_manual)
    def initUI(self):
        self.initMenu()
        self.main_widget = MainWidget(parent=self)
        self.setCentralWidget(self.main_widget)
        ## Focus
        self.main_widget.agenthandler.combo_box.clearFocus()
        self.setFocusPolicy(Qt.StrongFocus)

        # show
        self.setWindowTitle("Test")
        self.setGeometry(300, 300, 500, 800)
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
    def initState(self):
        self.gamestate = GameState()
        self.gamestate.boardstate = self.main_widget.main_board.boardstate.copy()
    def initAgent(self, is_manual):
        self.is_manual = is_manual
        self.main_widget.agenthandler.agentActionSignal.connect(self.agentActionHandler)
        self.main_widget.agenthandler.agentChangeSignal.connect(self.agentChangeSignalHandler)
    def keyPressEvent(self, a0):
        self.main_widget.keyPressEvent(a0)
        self.gamestate.boardstate = self.main_widget.main_board.boardstate.copy()
        self.gamestate.gameoverflag = self.main_widget.main_board.isGameOver()
        if self.main_widget.main_board.isGameOver():
            print("GameOver!")
            
    
    def agentActionHandler(self, event):
        if not self.is_manual:
            self.keyPressEvent(QKeyEvent(QEvent.KeyPress, event, Qt.NoModifier))
    
    def agentChangeSignalHandler(self, is_manual):
        if self.is_manual != is_manual:
            self.is_manual = is_manual 
        if self.is_manual:
            BlockUnit.animate_time = 100
        else:
            BlockUnit.animate_time = 1
            
            
class MainWidget(QWidget):
    def __init__(self, parent=None, is_manual=True):
        super().__init__()
        self.setParent(parent)
        self.initUI()
    def initUI(self):
        self.main_board = MainBoard(parent=self)
        self.agenthandler = AgentHandler(parent=self)
        self.main_board.move(0, 0)
        self.agenthandler.move(0, 500)
        
        # score change signal handling
        self.main_board.scoreChangeSig.connect(self.scoreChangeSignalHandler)
        
        
        


    
        
    def keyPressEvent(self, event):
        self.main_board.keyPressEvent(event)

    def scoreChangeSignalHandler(self, flag):
        if flag == 0:
            return self.main_board.getScore()
        elif flag == -1: ## game over
            pass
        elif flag == 1: ## you win
            pass
    

    
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GameApp()
    sys.exit(app.exec_())