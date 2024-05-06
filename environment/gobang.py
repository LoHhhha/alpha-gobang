import math

import numpy as np
from typing import Tuple
from environment.env import env


class game(env):
    # self.board: 0=None 1=you -1=op
    N = 0
    A = 1
    B = -1

    def __init__(self, board_size=15, win_size=5):
        super(game, self).__init__()
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.board_size = board_size
        self.win_size = win_size
        self.count = board_size * board_size
        self.draw_play = 19528  # 0x4c48 :)

    def get_neighbor_info(self, place: Tuple[int, int], agent: int):
        horizontal_count = int(self.board[place[0]][place[1]] == agent)
        for c in range(place[1] - 1, max(place[1] - self.win_size, -1), -1):
            if self.board[place[0]][c] != agent:
                break
            else:
                horizontal_count += 1
        # check up win_size-1 place
        for c in range(place[1] + 1, min(self.board_size, place[1] + self.win_size)):
            if self.board[place[0]][c] != agent:
                break
            else:
                horizontal_count += 1

        vertical_count = int(self.board[place[0]][place[1]] == agent)
        for r in range(place[0] - 1, max(place[0] - self.win_size, -1), -1):
            if self.board[r][place[1]] != agent:
                break
            else:
                vertical_count += 1
        for r in range(place[0] + 1, min(self.board_size, place[0] + self.win_size)):
            if self.board[r][place[1]] != agent:
                break
            else:
                vertical_count += 1

        diagonal_count = int(self.board[place[0]][place[1]] == agent)
        for d in range(1, min(min(place[0], place[1]) + 1, self.win_size)):
            if self.board[place[0] - d][place[1] - d] != agent:
                break
            else:
                diagonal_count += 1

        for d in range(1, min(min(self.board_size - place[0], self.board_size - place[1]), self.win_size)):
            if self.board[place[0] + d][place[1] + d] != agent:
                break
            else:
                diagonal_count += 1

        reverse_diagonal_count = int(self.board[place[0]][place[1]] == agent)
        for d in range(1, min(min(place[0] + 1, self.board_size - place[1]), self.win_size)):
            if self.board[place[0] - d][place[1] + d] != agent:
                break
            else:
                reverse_diagonal_count += 1

        for d in range(1, min(min(self.board_size - place[0], place[1] + 1), self.win_size)):
            if self.board[place[0] + d][place[1] - d] != agent:
                break
            else:
                reverse_diagonal_count += 1

        return horizontal_count, vertical_count, diagonal_count, reverse_diagonal_count

    def check(self) -> int:
        if self.pre_action is None:
            return 0

        action = self.pre_action

        self_color = self.board[action[0]][action[1]]
        horizontal_count, vertical_count, diagonal_count, reverse_diagonal_count = \
            self.get_neighbor_info(
                action,
                self_color
            )

        done = self.board[action[0]][action[1]] if max(
            horizontal_count,
            vertical_count,
            diagonal_count,
            reverse_diagonal_count,
        ) >= self.win_size else 0

        if done == 0 and self.count == 0:
            return self.draw_play
        return done

    def get_reward(self):
        if self.pre_action is None:
            # this is using to let module learn how to select place
            return -2560
        action = self.pre_action

        self_color = self.board[action[0]][action[1]]

        reward = 0

        # blank
        # guide to keep center place
        horizontal_count, vertical_count, diagonal_count, reverse_diagonal_count = \
            self.get_neighbor_info(
                action,
                self.N
            )
        reward += math.pow(math.e,
                           (horizontal_count + vertical_count + diagonal_count + reverse_diagonal_count - 4)) * 10

        # self
        # guide to attack
        horizontal_count, vertical_count, diagonal_count, reverse_diagonal_count = \
            self.get_neighbor_info(
                action,
                self_color
            )
        reward += math.pow(math.e,
                           (horizontal_count + vertical_count + diagonal_count + reverse_diagonal_count - 4)) * 50

        if max(horizontal_count, vertical_count, diagonal_count, reverse_diagonal_count) >= self.win_size:
            return 2560

        # other
        # guide to defend
        horizontal_count, vertical_count, diagonal_count, reverse_diagonal_count = \
            self.get_neighbor_info(
                action,
                self.A if self_color == self.B else self.B
            )
        reward += math.pow(math.e,
                           (horizontal_count + vertical_count + diagonal_count + reverse_diagonal_count - 4)) * 100

        return reward

    def step(self, agent: int, action: Tuple[int, int]) -> None:
        if self.board[action[0]][action[1]] != self.N:
            self.pre_action = None
            return
        self.count -= 1
        self.board[action[0]][action[1]] = agent
        self.pre_action = action

    def get_state(self, agent):
        state = self.board.copy().reshape(self.board_size * self.board_size)
        for i in range(len(state)):
            if state[i] != self.N:
                state[i] = (self.A if state[i] == agent else self.B)
        return state

    def clear(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.pre_action = None
        self.count = self.board_size * self.board_size

    def display(self):
        disp = ".OX"
        for i in range(self.board_size):
            for j in range(self.board_size):
                print(disp[self.board[i][j]], end=" ")
            print()
        print(f"{disp[self.A]}:A, {disp[self.B]}:B")


if __name__ == '__main__':
    gobang = game(board_size=15, win_size=5)
    while gobang.check() == 0:
        gobang.display()
        x, y = input().split(' ')
        x, y = int(x), int(y)
        gobang.step(agent=1, action=(x, y))
        print(gobang.get_reward())
