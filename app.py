import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget, QMessageBox, QPushButton
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtCore import Qt

from PyQt5.QtCore import Qt, QEvent


from debugtools import *

from gameState import GameState


from board import MainBoard, BlockUnit
from agenthandler import AgentHandler

from PyQt5.QtCore import pyqtSignal
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.05
EPS_END = 0.9
EPS_DECAY = 500
TAU = 0.005
LR = 1e-4
class GameApp(QMainWindow):
    stepDoneSignal = pyqtSignal(int) # 0: 문제 없음. 1: 문제 있음
    def __init__(self, is_manual=True):
        super().__init__()
        self.main_widget = None
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
        ## show
        self.setWindowTitle("2048!")
        self.setGeometry(300, 300, 600, 800)
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
        assert self.main_widget is not None, "\
            Implement Error! initUI() must be called before call initState()"
        self.gamestate = GameState()
        self.gamestate.boardstate = self.main_widget.main_board.boardstate.copy()
        self.gamestate.prev_boardstate = self.main_widget.main_board.prev_boardstate.copy()
        self.gamestate.able_actions = self.main_widget.main_board.able_actions.copy()
        self.gamestate.n_merged_blocks = 0
        self.gamestate.score = 0
    def initAgent(self, is_manual):
        assert self.main_widget is not None, "\
            Implement Error! initUI() must be called before call initAgent()"
        self.is_manual = is_manual
        self.is_train = False
        self.main_widget.agenthandler.agentActionSignal.connect(self.agentActionHandler)
        self.main_widget.agenthandler.agentChangeSignal.connect(self.agentChangeSignalHandler)
        self.main_widget.agenthandler.trainModeChangeSignal.connect(self.trainModeSignalHandler)

    def keyPressEvent(self, a0):
        if a0.key() not in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]:
            return
        ## update board
        self.main_widget.keyPressEvent(a0)
        ## update gamestate(used by agent)
        selected_action = [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D].index(a0.key())
        self.updateGameState(selected_action=selected_action)
        ## optimize model
        if self.is_train and not self.is_manual:
            self.main_widget.agenthandler.optimize()
        ## game over
        if self.main_widget.main_board.isGameOver():
            self.gameOverHandler()
    
    def gameOverHandler(self):
        assert self.main_widget.main_board.isGameOver(), "Game is not overed yet!"
        if self.is_train:
            import math
            with open("report.txt", "a") as f:
                iteration = self.main_widget.agenthandler.agent.iteration
                eps = self.main_widget.agenthandler.agent.eps
                f.write("\n" + f"iteration: {iteration}, score: {self.gamestate.score}, eps: {eps}")
            self.main_widget.resetGame()
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Game Over")
            msg_box.setText(f"Game Over!\nScore: {self.gamestate.score}\nboardstate: {self.gamestate.boardstate}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.buttonClicked.connect(app.quit)
            msg_box.exec_()
            sys.exit()
            print("GameOver!")
            print(self.gamestate)
            
        self.stepDoneSignal.emit(0)
         
            
    
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
    
    def trainModeSignalHandler(self, train_mode):
        if train_mode == 0: ## train
            self.is_train = True
            with open("report.txt", "w") as f:
                f.write("2048 RL!")
        else:
            self.is_train = False

    def updateGameState(self, selected_action):
        self.gamestate.boardstate = self.main_widget.main_board.boardstate.copy()
        self.gamestate.prev_boardstate = self.main_widget.main_board.prev_boardstate.copy()
        self.gamestate.prev_score = self.gamestate.score
        self.gamestate.score = self.main_widget.main_board.getScore()
        self.gamestate.selected_action = selected_action
        self.gamestate.gameoverflag = self.main_widget.main_board.isGameOver()
        self.gamestate.able_actions = self.main_widget.main_board.able_actions.copy()
        self.gamestate.n_merged_blocks = self.main_widget.main_board.n_merged_block
    def closeEvent(self, event):
        self.main_widget.closeEvent(event)
        event.accept()
class MainWidget(QWidget):
    def __init__(self, parent=None, is_manual=True):
        super().__init__()
        self.setParent(parent)
        self.initUI()
    def initUI(self):
        self.main_board = MainBoard(parent=self)
        self.agenthandler = AgentHandler(parent=self)
        self.reset_btn = QPushButton(self)
        self.reset_btn.setText("Reset")
        self.reset_btn.clicked.connect(self.resetGame)
        self.main_board.move(0, 0)
        self.agenthandler.move(0, 500)
        self.reset_btn.move(510, 0)
        
        # score change signal handling
        self.main_board.scoreChangeSig.connect(self.scoreChangeSignalHandler)
        
        
        


    def resetGame(self):
        self.main_board.reset()
        self.parent().initState()
        
    def keyPressEvent(self, event):
        self.main_board.keyPressEvent(event)

    def scoreChangeSignalHandler(self, flag):
        if flag == 0:
            return self.main_board.getScore()
        elif flag == -1: ## game over
            pass
        elif flag == 1: ## you win
            pass
    def closeEvent(self, event):
        self.main_board.closeEvent(event)
        self.agenthandler.closeEvent(event)

    
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GameApp()
    sys.exit(app.exec_())