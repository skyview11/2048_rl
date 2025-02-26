import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPainter, QColor
class Animator:
    def __init__(self, qobj):
        print("HAHAHA")
        print(qobj.rectX)
        self.qobj = qobj
        self.animation = QPropertyAnimation(qobj, b"rectX")
        self.animation.setDuration(2000)  # 2초 동안 애니메이션
        self.animation.setStartValue(50)
        self.animation.setEndValue(450)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.setLoopCount(-1)  # 무한 반복
        # self.animation.start()
    

    
class MovingRectangleWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Moving Rectangle Animation")
        self.setGeometry(100, 100, 600, 400)
        self._rect_x = 50  # 내부 변수로 변경
        self.rect_y = 150
        self.rect_width = 100
        self.rect_height = 50

        # 애니메이션 설정
        setattr(self, "rectX", pyqtProperty(int, self.get_rect_x, self.set_rect_x))  # pyqtProperty로 등록
        self.animator = Animator(self)
        self.animator.animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(100, 100, 255))  # 파란색 직사각형
        painter.drawRect(self._rect_x, self.rect_y, self.rect_width, self.rect_height)
    
    def get_rect_x(self):
        return self._rect_x

    def set_rect_x(self, value):
        self._rect_x = value
        self.update()  # 다시 그리기 요청
    # rectX = pyqtProperty(int, get_rect_x, set_rect_x)  # pyqtProperty로 등록
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MovingRectangleWidget()
    window.show()
    sys.exit(app.exec())
