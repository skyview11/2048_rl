import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from debugtools import *

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
        self.main_widget.keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())