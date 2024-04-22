from torch import nn


class test_demo(nn.Module):
    def __init__(self, state_size, board_size):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(state_size, state_size * 2),
            nn.Linear(state_size * 2, state_size),
            nn.Linear(state_size, board_size * board_size),
            nn.ReLU()
        )
        self.board_size = board_size
        self.state_size = state_size

    def forward(self, data):
        data = self.nn(data)
        return data
