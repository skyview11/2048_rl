from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel
from PyQt5.QtWidgets import QRadioButton, QButtonGroup

from PyQt5.QtCore import pyqtSignal, Qt, QTimer

from agent.manualAgent import ManualAgent
from agent.randomAgent import RandomAgent
from agent.DQNAgent import DQNAgent
class AgentHandler(QWidget):
    agentChangeSignal = pyqtSignal(int) ## is_manual 을 전송(일단은?)
    agentActionSignal = pyqtSignal(Qt.Key)
    trainModeChangeSignal = pyqtSignal(int) # 0: train, 1: test
    def __init__(self, parent=None, update_freq = 1):
        super().__init__()
        self.agents = ["manual control", "random agent", "dqn"]
        self.is_timer_running = False
        self.timer_update_freq = update_freq
        self.agent = ManualAgent()
        self.setParent(parent)
        self.init_ui()

    def init_ui(self):
        self.resize(500, 200)
        layout = QVBoxLayout()
        
        ## combo box
        self.combo_box = QComboBox()
        self.combo_box.setMinimumSize(500, 50)
        self.combo_box.setStyleSheet("font-size: 16px;")
        self.combo_box.addItems(self.agents)
        self.combo_box.currentIndexChanged.connect(self.update_agent)
        layout.addWidget(self.combo_box)
        
        ## train/test button
        self.is_train = False
        self.train_mode_btn = QRadioButton("train", self)
        self.test_mode_btn = QRadioButton("test", self)
        self.test_mode_btn.setChecked(True)
        
        self.train_mode_btn_group = QButtonGroup(self)
        self.train_mode_btn_group.addButton(self.train_mode_btn)
        self.train_mode_btn_group.addButton(self.test_mode_btn)
        
        self.train_mode_btn_group.buttonClicked.connect(self.change_train_mode)
        layout.addWidget(self.train_mode_btn)
        layout.addWidget(self.test_mode_btn)
        self.setLayout(layout)
        self.parent().parent().stepDoneSignal.connect(self.optimize)

    def update_agent(self, index):
        agent_str = self.agents[index]
        print(agent_str)
        if self.is_timer_running:
            self.timer.stop()
            self.is_timer_running = False
        if  agent_str == "manual control":
            self.agentChangeSignal.emit(1)
            return
        elif agent_str == "random agent":
            self.agentChangeSignal.emit(0)
            self.agent = RandomAgent()
        elif agent_str == "dqn":
            self.agentChangeSignal.emit(0)
            self.agent = DQNAgent()
        # self.timer = MicrosecondTimer(self, self.timer_update_freq)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.act)
        self.is_timer_running = True
        self.timer.start(self.timer_update_freq)
    def keyId2Key(self, id):
        id_map = [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]
        return id_map[id]
    def act(self):
        gamestate = self.parent().parent().gamestate
        action = self.agent.policy(gamestate, self.is_train)
        self.agentActionSignal.emit(action)
    
    def change_train_mode(self, button):
        if button.text() == "train":
            self.is_train = 1
            self.trainModeChangeSignal.emit(0)
        else:
            self.is_train = 0
            self.trainModeChangeSignal.emit(1)
    
    def optimize(self):
        gamestate = self.parent().parent().gamestate
        self.agent.optimize(gamestate)

    def closeEvent(self, event):
        self.closed = True
        self.agent.close()
        event.accept()

from PyQt5.QtCore import QThread, pyqtSignal
import time

class MicrosecondTimer(QThread):
    timeout = pyqtSignal()  # 타이머 신호

    def __init__(self, parent, interval_us):
        super().__init__()
        # self.parent() = parent
        self.interval = interval_us / 1_000_000  # 마이크로초 → 초 변환
        self.running = True

    
    def run(self):
        while self.running:
            start_time = time.perf_counter()
            self.timeout.emit()  # 신호 발생
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False

    

if __name__ == "__main__":
    app = QApplication([])
    window = AgentHandler()
    window.show()
    app.exec_() 
