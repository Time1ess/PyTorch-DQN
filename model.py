from random import sample, random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable


class ReplayMemory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.idx = 0
        self.memory = []

    def sample(self, batch_size):
        indices = sample(range(1, len(self)), batch_size)
        batch = []
        obs = []
        for idx in indices:
            obs.append(self.memory[idx-1][2])
            batch.append(self.memory[idx])
        acts, rews, next_obs, finals = list(zip(*batch))
        return obs, acts, rews, next_obs, finals

    def reset(self):
        self.memory.clear()
        self.idx = 0

    def push(self, transition):
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.memory_size

    def __len__(self):
        return len(self.memory)


class History(object):
    def __init__(self, history_size):
        self.history_size = history_size
        self.reset()

    def push(self, x):
        self.history[1:, :, :] = self.history[:-1, :, :]
        self.history[0, :, :] = x

    def reset(self):
        self.history = np.zeros((4, 84, 84)).astype(float)

    def get_history(self):
        return torch.from_numpy(
            self.history.reshape((1, 4, 84, 84))).type(torch.FloatTensor)


class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, output_size)

        # Init
        self.conv1.weight.data.normal_(std=0.01)
        self.conv1.bias.data.zero_().add_(0.1)
        self.conv2.weight.data.normal_(std=0.01)
        self.conv2.bias.data.zero_().add_(0.1)
        self.fc1.weight.data.normal_(std=0.01)
        self.fc1.bias.data.zero_().add_(0.1)
        self.fc2.weight.data.normal_(std=0.01)
        self.fc2.bias.data.zero_().add_(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 2592)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DQNAgent(object):
    def __init__(self, env, output_size, args):
        self.args = args
        self.env = env
        self.memory = ReplayMemory(args.memory_size)
        self.history = History(args.history_size)
        self.steps = 0

        self.dqn = DQN(output_size)
        self.target_dqn = DQN(output_size)
        self.criterion = nn.MSELoss()
        if args.use_cuda:
            self.dqn = self.dqn.cuda()
            self.target_dqn = self.target_dqn.cuda()
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=args.lr,
                                       momentum=args.g_momentum)
        self.train()

    def train(self, mode=True):
        self.train_mode = mode

    def q_learn_mini_batch(self):
        args = self.args
        obs, acts, rews, next_obs, finals = self.sample_memory(args.batch_size)
        observations = torch.cat(obs).type(args.FloatTensor)
        actions = Variable(args.LongTensor(acts).view(-1, 1))
        rewards = Variable(args.FloatTensor(rews))
        next_observations = torch.cat(next_obs).type(args.FloatTensor)
        game_over_mask = args.ByteTensor(finals)
        next_rewards = self.qvalue(
            Variable(next_observations, volatile=True),
            use_target=True).max(1)[0]
        next_rewards[game_over_mask] = 0
        target_rewards = rewards + args.gamma * next_rewards
        target_rewards.volatile = False
        prediction_rewards = self.qvalue(
            Variable(observations)).gather(1, actions)
        loss = self.criterion(prediction_rewards, target_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def perceive(self, observation, action, reward, done):
        self.push_observation(observation)
        new_history = self.get_history()
        if self.train_mode:
            self.memory.push((action, reward, new_history, done))

        if self.train_mode and self.steps > self.args.learn_start:
            self.q_learn_mini_batch()

    def qvalue(self, x, use_target=False):
        if use_target:
            return self.target_dqn(x)
        else:
            return self.dqn(x)

    def predict(self, history, test=None):
        steps = self.steps
        ep = test or max(0.1, self.args.epsilon * (1 - steps / 1e6))

        if test is None and steps > self.args.learn_start and random() >= ep:
            var_his = Variable(history.cuda(), volatile=True)
            action = self.target_dqn(var_his).max(1)[1].cpu().data[0]
        else:
            action = self.env.action_space.sample()
        self.steps += 1
        return action

    def update_target_dqn(self):
        for param, target_param in zip(self.dqn.parameters(),
                                       self.target_dqn.parameters()):
            target_param.data = param.data.clone()

    def save(self, episodes):
        torch.save(self.dqn.state_dict(),
                   'saves/DQN_{}_{}.bin'.format(self.args.env_name, episodes))

    def reset_history(self):
        self.history.reset()

    def push_observation(self, observation, repeat=1):
        for _ in range(repeat):
            self.history.push(observation)

    def get_history(self):
        return self.history.get_history()

    def sample_memory(self, batch_size):
        return self.memory.sample(batch_size)

    @property
    def memory_size(self):
        return len(self.memory)
