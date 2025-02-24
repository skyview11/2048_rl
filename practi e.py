import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIcon

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
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout()
        main_widget.setLayout(vbox)
        
        vbox.addWidget(QLabel("A:"), 0)
        vbox.addWidget(QLabel("B:"), 1)
        vbox.addWidget(QLabel("C:"), 2)
        
        # show
        self.setWindowTitle("Test")
        self.setGeometry(300, 300, 800, 800)
        self.show()
        
    def gameBoardWIdget(self):
        pass
    def initMenu(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        ## file menu
        filemenu = menubar.addMenu('&File')
        exit_action = QAction("Exit", self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(qApp.quit)
        
        filemenu.addAction(exit_action)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())