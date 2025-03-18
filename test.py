import torch
from agent.DQNAgent import QFunction


def main():
    q = QFunction().to("cuda")
    q.load_state_dict(torch.load("./ckpt/iter_3000k.pth", weights_only=True))
    boardstate = [[-1, -1, 2, 2], 
                  [-1, -1, -1, -1], 
                  [-1, -1, -1, -1],
                  [-1, 4, -1, -1]]
    boardstate = torch.tensor(boardstate,dtype=torch.float).flatten().to("cuda")
    print(q(boardstate))
    
if __name__ == "__main__":
    main()