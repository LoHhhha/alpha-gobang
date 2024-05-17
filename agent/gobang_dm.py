import queue
import random
import torch

import environment.gobang


class dm_robot:
    def __init__(self, color, env: environment.gobang.game, display_reward: bool = False):
        self.env = env
        self.board_size = env.board_size
        self.win_size = env.win_size
        self.color = color
        self.display_reward = display_reward

    def get_action(self, state):

        qu = queue.Queue()
        best_score = float('-inf')
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):

                if self.env.board[i][j] == self.env.N:
                    # temp to let board[i][j] to be self.color
                    self.env.board[i][j] = self.color
                    reward = self.env.get_reward((i, j))
                    self.env.board[i][j] = self.env.N

                    if reward > best_score:
                        best_score = reward
                        qu = queue.Queue()
                        qu.put(i * self.board_size + j)
                    elif reward == best_score:
                        qu.put(i * self.board_size + j)

                    if self.display_reward:
                        print(reward, end=" ")
                elif self.display_reward:
                    print("Prohibited", end=" ")
            if self.display_reward:
                print()

        save_p = 1.0
        best_chose = -1
        while not qu.empty():
            if random.random() <= save_p:
                best_chose = qu.get()
            else:
                qu.get()
            save_p *= 0.5

        action = torch.zeros(self.board_size * self.board_size, dtype=torch.float)
        action[best_chose] = 1
        return action

    def reduce_epsilon(self):
        pass
