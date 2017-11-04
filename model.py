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
        self.cnt = 0
        self.histories = np.empty((self.memory_size, 4, 84, 84),
                                  dtype=np.float32)
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.finals = np.empty(self.memory_size, dtype=np.uint8)

    def sample(self, batch_size):
        indices = sample(range(1, len(self)), batch_size)
        prev_ob_indices = [idx - 1 for idx in indices]
        histories = self.histories[prev_ob_indices]
        acts = self.actions[indices]
        rews = self.rewards[indices]
        next_histories = self.histories[indices]
        finals = self.finals[indices]
        return histories, acts, rews, next_histories, finals

    def reset(self):
        self.memory.clear()
        self.idx = 0

    def push(self, transition):
        action, reward, history, done = transition
        self.histories[self.idx, :, :] = history
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.finals[self.idx] = 1 if done else 0
        self.idx = (self.idx + 1) % self.memory_size
        self.cnt += 1

    def __len__(self):
        return min(self.cnt, self.memory_size)


class History(object):
    def __init__(self, history_size):
        self.history_size = history_size
        self.reset()

    def push(self, x):
        self.history[1:, :, :] = self.history[:-1, :, :]
        self.history[0, :, :] = x

    def reset(self):
        self.history = np.zeros((4, 84, 84), dtype=np.float32)

    def get_history(self):
        return self.history


class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)

        return x


class DQNAgent(object):
    def __init__(self, env, output_size, args):
        self.args = args
        self.env = env
        self.memory = ReplayMemory(args.memory_size)
        self.history = History(args.history_size)
        self.steps = 0
        self.epsilon = args.epsilon
        self.total_qvalue = 0
        self.update_count = 0

        self.dqn = DQN(output_size)
        self.target_dqn = DQN(output_size)
        self.criterion = nn.MSELoss()
        if args.use_cuda:
            self.dqn = self.dqn.cuda()
            self.target_dqn = self.target_dqn.cuda()
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=args.lr,
                                       alpha=args.alpha,
                                       eps=args.rmsprop_ep)
        self.train()

    def train(self, mode=True):
        self.train_mode = mode

    def q_learn_mini_batch(self):
        args = self.args
        obs, acts, rews, next_obs, finals = self.sample_memory(args.batch_size)
        observations = Variable(torch.from_numpy(obs).cuda())
        actions = Variable(torch.from_numpy(acts).view(-1, 1).long().cuda())
        rewards = Variable(torch.from_numpy(rews).cuda())
        next_observations = Variable(torch.from_numpy(next_obs).cuda(),
                                     volatile=True)
        game_over_mask = torch.from_numpy(finals).cuda()
        next_rewards = self.qvalue(next_observations,
                                   use_target=True).max(1)[0]
        next_rewards[game_over_mask] = 0
        target_rewards = (rewards + args.gamma * next_rewards).view(-1, 1)
        target_rewards.volatile = False
        prediction_rewards = self.qvalue(observations).gather(1, actions)
        bellman_errors = (prediction_rewards - target_rewards)
        clipped_errors = bellman_errors.clamp(-1, 1)
        self.optimizer.zero_grad()
        prediction_rewards.backward(clipped_errors.data)
        self.optimizer.step()
        self.total_qvalue += prediction_rewards.data.cpu().mean()
        self.update_count += 1
        if self.update_count % args.update_freq == 0:
            self.update_target_dqn()
            self.total_qvalue = 0
            self.update_count = 0

    def perceive(self, observation, action, reward, done):
        self.push_observation(observation)
        history = self.get_history()
        if self.train_mode:
            self.memory.push((action, reward, history, done))

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
            history = torch.from_numpy(history).view(1, 4, 84, 84).cuda()
            var_his = Variable(history, volatile=True)
            action = self.dqn(var_his).max(1)[1].cpu().data[0]
        else:
            action = self.env.action_space.sample()
        self.steps += 1
        self.epsilon = ep
        return action

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

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
