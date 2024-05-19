import queue
import random
import torch

import environment.gobang

import random


class random_robot():
    def __init__(self, color, env: environment.gobang.game, display_reward: bool = False):
        self.env = env
        self.board_size = env.board_size
        self.win_size = env.win_size
        self.color = color
        self.display_reward = display_reward

    def get_action(self, state):
        length = self.board_size * self.board_size
        random_num = random.randint(0, length)
        best_chose = -1
        while True:
            if state[random_num % length] == 0:
                best_chose = random_num % length
                break
            random_num += 1

        action = torch.zeros(self.board_size * self.board_size, dtype=torch.float)
        action[best_chose] = 1
        return action

    def reduce_epsilon(self):
        pass
