from dataclasses import dataclass, field
from typing import List
@dataclass
class GameState:
    boardstate: List[int] = field(default_factory=lambda:[-1]*16)
    score: int = 0
    gameoverflag: bool = False
    
if __name__ == "__main__":
    gs = GameState()
    gs.boardstate = [1]*16
    print(gs)