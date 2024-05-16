import torch
from torch import nn


class L_Net(nn.Module):
    def __init__(self, board_size):
        super().__init__()

        self.conv_size = [1, 8, 32, 64, 128]

        self.get_mask = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.catch = nn.Sequential(
            nn.Conv1d(
                in_channels=self.conv_size[0],
                out_channels=self.conv_size[1],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[1],
                out_channels=self.conv_size[2],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[2],
                out_channels=self.conv_size[2],
                stride=1, kernel_size=3, padding=1
            ),
            nn.PReLU(),
            nn.Conv1d(
                in_channels=self.conv_size[2],
                out_channels=self.conv_size[3],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[3],
                out_channels=self.conv_size[4],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[4],
                out_channels=self.conv_size[4],
                stride=1, kernel_size=3, padding=1
            ),
            nn.PReLU(),
        )

        self.push = nn.Sequential(
            nn.Conv1d(
                in_channels=self.conv_size[4],
                out_channels=self.conv_size[3],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Dropout(0.5),
            nn.Conv1d(
                in_channels=self.conv_size[3],
                out_channels=self.conv_size[2],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[2],
                out_channels=self.conv_size[2],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv1d(
                in_channels=self.conv_size[2],
                out_channels=self.conv_size[1],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[1],
                out_channels=self.conv_size[0],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Conv1d(
                in_channels=self.conv_size[0],
                out_channels=self.conv_size[0],
                stride=1, kernel_size=3, padding=1
            ),
            nn.Softmax(),
        )

        self.board_size = board_size

    def forward(self, state):
        res = None
        for s in state:
            current_state = s.unsqueeze(0)
            current_state = current_state + self.get_mask(current_state)

            feature = self.catch(current_state)
            if res is None:
                res = self.push(feature)
            else:
                res = torch.cat((res, self.push(feature)))
        return res


if __name__ == '__main__':
    _input = torch.zeros(1, 9)
    _input[0,6]=1
    net = L(board_size=9)
    output = net(_input)
    print(output.shape)
