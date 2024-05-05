import math

import torch
from torch import nn


class test_demo(nn.Module):
    def __init__(self, state_size, board_size):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(state_size, state_size),
            nn.Linear(state_size, state_size * 4),
            nn.PReLU(),
            nn.Linear(state_size * 4, state_size),
            nn.Softplus(),
            nn.Linear(state_size, board_size * board_size),
            nn.PReLU(),
            nn.Linear(board_size * board_size, board_size * board_size),
            nn.PReLU(),
        )
        self.board_size = board_size
        self.state_size = state_size

    def forward(self, data):
        data = self.nn(data)
        return data
