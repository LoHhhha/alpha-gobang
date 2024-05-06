import torch
from torch import nn


class hNet_RL_v1(nn.Module):
    def __init__(self, board_size):
        super().__init__()

        self.conv_size = [1,8,16,32,64]

        self.catch = nn.Sequential(
            nn.Conv1d(in_channels=self.conv_size[0], out_channels=self.conv_size[1], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[1], out_channels=self.conv_size[2], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[2], out_channels=self.conv_size[2], stride=1, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.conv_size[2], out_channels=self.conv_size[3], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[3], out_channels=self.conv_size[4], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[4], out_channels=self.conv_size[4], stride=1, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.push = nn.Sequential(
            nn.Conv1d(in_channels=self.conv_size[4], out_channels=self.conv_size[3], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[3], out_channels=self.conv_size[2], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[2], out_channels=self.conv_size[2], stride=1, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.conv_size[2], out_channels=self.conv_size[1], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[1], out_channels=self.conv_size[0], stride=1, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=self.conv_size[0], out_channels=self.conv_size[0], stride=1, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.board_size = board_size

    def forward(self, state):
        # `state` and `state` have the same size (board_size, board_size)
        # `mask` using to mask the action

        res = None
        for s in state:
            feature = self.catch(s.unsqueeze(0))
            if res is None:
                res = self.push(feature)
            else:
                res = torch.cat((res, self.push(feature)))
        return res


if __name__ == '__main__':
    _input = torch.randn(2, 9)
    net = hNet_RL_v1(board_size=9)
    output = net(_input)
    print(output.shape)
