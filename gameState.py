from dataclasses import dataclass, field
from typing import List
from PyQt5.QtCore import Qt
@dataclass
class GameState:
    boardstate: List[int] = field(default_factory=lambda:[-1]*16)
    prev_boardstate: List[int] = field(default_factory=lambda:[-1]*16)
    prev_score: int = 0
    score: int = 0
    selected_action = 0 # 0: w, 1: a, 2: s, 3: d
    gameoverflag: bool = False
    able_actions: List[Qt.Key] = field(default_factory=lambda:[])
    n_merged_blocks: int = 0
if __name__ == "__main__":
    gs = GameState()
    gs.boardstate = [1]*16
    print(gs)