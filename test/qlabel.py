import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # QLabel 생성
        label = QLabel("여기에 배치!", self)

        # 좌표 (x=50, y=100) 위치에 배치
        label.setGeometry(0, 0, 50, 50)

        # 윈도우 설정
        self.setWindowTitle("QLabel 위치 지정")
        self.setGeometry(100, 100, 400, 300)  # (x, y, width, height)
        
        label.setStyleSheet("background-color: lightblue; color: black; font-size: 18px; padding: 10px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
