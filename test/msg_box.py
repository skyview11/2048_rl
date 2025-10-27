import sys
from PyQt5.QtWidgets import QApplication, QMessageBox

def show_game_over():
    app = QApplication(sys.argv)
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Game Over")
    msg_box.setText("게임 종료!")
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()

if __name__ == "__main__":
    show_game_over()
