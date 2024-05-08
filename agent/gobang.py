import math
from collections import deque
import random
import torch.nn
import numpy as np
from agent.dqn_trainer import DQN

# if you want to change net, plz just "... as Net"
from agent.module.hNet_RL_v1 import hNet_RL_v1 as Net

MAX_MEMORY = 10240
BATCH_SIZE = 1024


class robot(DQN):
    def __init__(self,
                 module_save_path: str | None = None,
                 device: torch.device = torch.device('cpu'),
                 epsilon=0.4,
                 epsilon_decay=0.99,
                 board_size=15,
                 lr=0.01,
                 max_memory_size=MAX_MEMORY,
                 batch_size=BATCH_SIZE,
                 ):
        super().__init__(learning_rate=lr)

        if module_save_path is None:
            self.module = Net(board_size=board_size)
        else:
            self.module = torch.load(module_save_path)

        self.module = self.module.to(device)

        self.optimizer = self.optimizer(lr=self.learning_rate, params=self.module.parameters())
        self.loss = self.loss()

        self.device = device
        self.memory = deque(maxlen=max_memory_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size

        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        self.loss = torch.nn.SmoothL1Loss()

    def change_module(self, module_save_path: str):
        self.module = torch.load(module_save_path)
        self.module = self.module.to(self.device)

    def change_module_from_other(self, other):
        state = other.module.state_dict()
        self.module.load_state_dict(state)

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
        # update:
        # let robot always make a logical action.
        # base action will not affect train, so let the place chosen be 1.
        if random.random() < self.epsilon or need_random:
            # random chosen
            action = self.random_action(state)
        else:
            action = self.module(torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0))[0].detach()
            # print(action)
            best_place = -1
            best_score = float('-inf')
            for i in range(len(state)):
                if state[i] == 0:
                    if best_score < action[i]:
                        best_score = action[i]
                        best_place = i
            action = torch.zeros(self.board_size * self.board_size, dtype=torch.float)
            action[best_place] = 1
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        d_next_state = next_state.to(self.device)
        d_reward = reward.to(self.device)
        d_state = state.to(self.device)

        pred = self.module(d_state)

        # think now and future
        # the module predicts the reward where the action will get
        # Q(s, a) = r + gamma * max{Q(s', a')}
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = d_reward[idx]
            if done[idx] == 0:
                next_pre = self.module(d_next_state[idx].unsqueeze(0))
                # do not let other get the best
                Q_new -= self.gamma * torch.max(next_pre)
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # change target
        for i in range(len(done)):
            for j in range(len(state[i])):
                # cannot place
                if state[i][j] != 0:
                    target[i][j] = -2560

        self.optimizer.zero_grad()
        loss = self.loss(pred, target)
        loss.backward()
        self.optimizer.step()

    def train_action(self, state, action, reward, next_state, done):
        self.train(state, action, reward, next_state, done)

    def train_memory(self):
        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.train(states, actions, rewards, next_states, dones)

    def save(self, path: str):
        torch.save(self.module, path)
