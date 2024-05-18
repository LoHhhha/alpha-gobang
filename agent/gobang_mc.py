from collections import deque
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn

import environment
# if you want to change net, plz just "... as Net"
from agent.module.hNet_RL_v1 import hNet_RL_v1_Sigmoid as Net

MAX_MEMORY = 10240
BATCH_SIZE = 1024


class mc_robot:
    def __init__(
            self,
            learning_rate=0.01,
            board_size=15,
            search_node_number=10,
            small_random_select_rate=0.3,
            gamma=0.6,
            draw_play_is_win=False,
            value_from_dm=False,  # true: node sort by dmt, else sort by module output
            max_memory_size=MAX_MEMORY,
            batch_size=BATCH_SIZE,
            loss_class=torch.nn.MSELoss,
            optimizer_class=torch.optim.Adam,
            module: torch.nn.Module = None,
            device: torch.device = torch.device('cpu'),
    ):
        self.learning_rate = learning_rate
        if module is None:
            self.module = Net(board_size=board_size)
            for m in self.module.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
        else:
            self.module = module
        self.module = self.module.to(device)
        self.loss = loss_class()
        self.loss = self.loss.to(device)

        self.optimizer = optimizer_class(lr=self.learning_rate, params=self.module.parameters())

        self.device = device

        self.board_size = board_size
        self.memory = deque(maxlen=max_memory_size)
        self.board_size = board_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.search_node_number = search_node_number
        self.small_random_select_number = int(small_random_select_rate * search_node_number)
        self.big_select_number = search_node_number - self.small_random_select_number
        self.value_from_dm = value_from_dm
        self.draw_play_is_win = draw_play_is_win

    def search_and_get_experience(self, env: environment.gobang, who: int) -> Tuple[int, int, int]:
        current_state = env.get_state(who)

        d_current_state = torch.Tensor(current_state).to(self.device).unsqueeze(0)
        current_output = self.module(d_current_state)[0].detach()

        next_place = []
        expected_output = np.zeros(self.board_size * self.board_size)
        for i in range(len(current_state)):
            if env.board[i // self.board_size][i % self.board_size] != env.N:
                expected_output[i] = 0
            else:
                expected_output[i] = current_output[i]
                if self.value_from_dm:
                    br, bc = i // self.board_size, i % self.board_size
                    env.board[br][bc] = who
                    next_place.append((i, env.get_reward((br, bc))))
                    env.board[br][bc] = env.N
                else:
                    next_place.append((i, current_output[i].item()))

        next_place = sorted(next_place, key=lambda x: x[1], reverse=True)

        if len(next_place) > self.search_node_number:
            next_place = next_place[:self.big_select_number] + \
                         random.sample(
                             next_place[len(next_place) - self.small_random_select_number:],
                             self.small_random_select_number
                         )
        win_leave_cnt, loss_leave_cnt, leave_cnt = 0, 0, 0
        for i, _ in next_place:
            place_r, place_c = i // self.board_size, i % self.board_size
            env.board[place_r][place_c] = who
            done = env.check((place_r, place_c))
            if done != 0:
                leave_cnt += 1
                if done == env.draw_play:
                    expected_output[i] = 1 if self.draw_play_is_win else 0
                    win_leave_cnt += 1 if self.draw_play_is_win else 0
                else:
                    expected_output[i] = 1
                    win_leave_cnt += 1
            else:
                next_agent = env.A if who == env.B else env.B
                sub_node_win_leave_cnt, sub_loss_leave_cnt, sub_node_leave_cnt = \
                    self.search_and_get_experience(env, next_agent)

                # tips: need convert
                leave_cnt += sub_node_leave_cnt
                if self.draw_play_is_win:
                    win_leave_cnt += sub_node_leave_cnt - sub_loss_leave_cnt
                    loss_leave_cnt += sub_node_win_leave_cnt
                else:
                    win_leave_cnt += sub_loss_leave_cnt
                    loss_leave_cnt += sub_node_leave_cnt - sub_node_win_leave_cnt

                if sub_node_leave_cnt == 0:
                    # draw
                    expected_output[i] = 1 if self.draw_play_is_win else 0
                else:
                    expected_output[i] = \
                        expected_output[i] * (1 - self.gamma) + sub_loss_leave_cnt * self.gamma / sub_node_leave_cnt

            env.board[place_r][place_c] = env.N

        self.train_action(current_state, expected_output)

        return win_leave_cnt, loss_leave_cnt, leave_cnt

    def get_action(self, state, show_result=False):
        action = self.module(torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0))[0].detach()
        if show_result:
            print(action)
        best_p, chosen = -1, -1
        base = 1.0
        for i in range(len(state)):
            if state[i] == 0:
                if best_p < action[i].item():
                    best_p = action[i].item()
                    chosen = i
                    base = 1
                elif best_p == action[i].item():
                    base *= 0.5
                    if random.random() <= base:
                        chosen = i
            action[i] = 0
        action[chosen] = 1
        return action

    def train(self, state, expected_output):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        expected_output = torch.tensor(np.array(expected_output), dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            expected_output = torch.unsqueeze(expected_output, 0)

        current_output = self.module(state)
        self.optimizer.zero_grad()
        loss = self.loss(current_output, expected_output)
        loss.backward()
        self.optimizer.step()

    def memorize(self, state, expected_output):
        self.memory.append((state, expected_output))

    def train_action(self, state, expected_output):
        self.memorize(state, expected_output)
        self.train(state, expected_output)

    def train_memory(self):
        if len(self.memory) < self.batch_size:
            states, expected_outputs = zip(*self.memory)
            self.train(states, expected_outputs)
            return

        sample = random.sample(self.memory, self.batch_size)
        states, expected_outputs = zip(*sample)
        self.train(states, expected_outputs)
