from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import pyqtSignal, Qt, QTimer

from agent.manualAgent import ManualAgent
from agent.randomAgent import RandomAgent
class AgentHandler(QWidget):
    agentChangeSignal = pyqtSignal(int) ## is_manual 을 전송(일단은?)
    agentActionSignal = pyqtSignal(Qt.Key)
    def __init__(self, parent=None, update_freq = 100):
        super().__init__()
        self.agents = ["manual control", "random agent"]
        self.is_timer_running = False
        self.timer_update_freq = update_freq
        self.agent = ManualAgent()
        self.setParent(parent)
        self.init_ui()

    def init_ui(self):
        self.resize(500, 50)
        layout = QVBoxLayout()
        self.combo_box = QComboBox()
        self.combo_box.setMinimumSize(500, 50)
        self.combo_box.setStyleSheet("font-size: 16px;")
        self.combo_box.addItems(self.agents)
        self.combo_box.currentIndexChanged.connect(self.update_agent)
        layout.addWidget(self.combo_box)
        self.setLayout(layout)

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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.act)
        self.is_timer_running = True
        self.timer.start(self.timer_update_freq)
    def keyId2Key(self, id):
        id_map = [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]
        return id_map[id]
    def act(self):
        action = self.agent.policy()
        self.agentActionSignal.emit(action)
        
        
if __name__ == "__main__":
    app = QApplication([])
    window = AgentHandler()
    window.show()
    app.exec_() 
