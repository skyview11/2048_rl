from PySide2.QtWidgets import QWidget
from PySide2.QtGui import QPalette, QPainter, QPen, QBrush, QFont
from PySide2.QtCore import Qt, QRect

class RenderArea(QWidget):
    def __init__(self,parent=None):
        QWidget.__init__(self,parent)
        self.setAutoFillBackground(True)   # 배경을 칠하도록 설정
        self.setBackgroundRole(QPalette.Base) # 배경색 QPalette.Base로 설정
    def paintEvent(self,event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)

        painter.setPen(QPen(Qt.black,4,Qt.DotLine,Qt.RoundCap))  # 펜 설절
        painter.setBrush(QBrush(Qt.green,Qt.SolidPattern)) # 브러시 설정
        painter.setFont(QFont("Arial",30))

        rect = QRect(80,80,400,200)
        painter.drawRoundRect(rect,50,50)  # 경계선은 펜, 내부는 브러시

        painter.drawText(rect,Qt.AlignCenter,"Hello, Qt!")  # 펜의 색상, 폰트 사용

from PySide2.QtWidgets import QApplication 
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)    

    renderArea = RenderArea()
    renderArea.setWindowTitle("Render Minimal")
    renderArea.resize(530,360)
    renderArea.show()

    app.exec_()