from collections import deque
import random
from agent.module.gobang import test_demo
from agent.dqn_trainer import DQN

import torch.nn

MAX_MEMORY = 10240
BATCH_SIZE = 512


class robot(DQN):
    def __init__(self,
                 module_save_path: str | None = None,
                 device: torch.device = torch.device('cpu'),
                 epsilon=0.4,
                 epsilon_decay=0.95,
                 board_size=15):
        super().__init__()

        if module_save_path is None:
            self.module = test_demo(state_size=board_size * board_size, board_size=board_size)

        else:
            self.module = torch.load(module_save_path)
        self.module = self.module.to(device)

        self.optimizer = self.optimizer(lr=self.learning_rate, params=self.module.parameters())
        self.loss = self.loss()

        self.device = device
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size

    def change_module(self, module_save_path: str):
        self.module = torch.load(module_save_path)
        self.module = self.module.to(self.device)

    def reduce_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def random_action(self, state):
        action = torch.zeros(self.board_size * self.board_size, dtype=torch.float)
        chosen = -1
        base = 1
        for i in range(len(state)):
            if state[i] == 0:
                if random.random() <= base:
                    base /= 2
                    chosen = i
        action[chosen] = 1
        return action.to(self.device)

    def get_action(self, state, need_random=False):
        if random.random() < self.epsilon or need_random:
            # random chosen
            action = self.random_action(state)
        else:
            action = self.module(torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0))[0].detach()
        return action

    def get_really_action(self, action):
        return torch.argmax(action[0:self.board_size]), torch.argmax(action[self.board_size:-1])

    def train_action(self, state, action, reward, next_state, done):
        self.train(state, action, reward, next_state, done)

    def train_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.train(states, actions, rewards, next_states, dones)

    def save(self, path: str):
        torch.save(self.module, path)
