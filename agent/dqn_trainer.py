import torch


class DQN:
    def __init__(self,
                 gamma=0.9,
                 learning_rate=0.01
                 ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.loss = torch.nn.MSELoss

        self.device = None
        self.module = None
        self.optimizer = torch.optim.Adam

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

        self.optimizer.zero_grad()
        loss = self.loss(pred, target)
        loss.backward()
        self.optimizer.step()
