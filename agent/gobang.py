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
                 epsilon_decay=0.99,
                 board_size=15,
                 lr=0.01
                 ):
        super().__init__(learning_rate=lr)

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

        self.loss = torch.nn.CrossEntropyLoss()

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
        # update:
        # let robot always make a logical action.
        if random.random() < self.epsilon or need_random:
            # random chosen
            action = self.random_action(state)
        else:
            action = self.module(torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0))[0].detach()
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
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

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
                Q_new += self.gamma * torch.max(self.module(d_next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # change target
        for i in range(len(done)):
            for j in range(len(state[i])):
                # cannot place
                if state[i][j] != 0:
                    target[i][j] = 0
        target = torch.nn.Softmax()(target.to(self.device))

        self.optimizer.zero_grad()
        loss = self.loss(pred, target)
        loss.backward()
        self.optimizer.step()

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
