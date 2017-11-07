from random import randint, random

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable


class ReplayMemory(object):
    def __init__(self, memory_size, history_size, batch_size):
        self.memory_size = memory_size
        self.history_size = history_size
        self.batch_size = batch_size
        self.idx = 0
        self.cnt = 0
        self.observations = np.empty((self.memory_size, 84, 84),
                                     dtype=np.uint8)
        print('Pre-allocating RAM')
        # Allocate memory immediately
        self.observations += 1
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.finals = np.empty(self.memory_size, dtype=np.uint8)

        self.prestates = np.empty([batch_size, history_size, 84, 84],
                                  dtype=np.float32)
        self.poststates = np.empty([batch_size, history_size, 84, 84],
                                   dtype=np.float32)

    def sample(self):
        indices = []
        while len(indices) < self.batch_size:
            while True:
                idx = randint(self.history_size, self.cnt - 1)
                # [new, idx, old], skip
                if idx >= self.idx and idx - self.history_size < self.idx:
                    continue
                # Not continuous frames, skip
                if self.finals[(idx - self.history_size):idx].any():
                    continue
                # Already sampled, skip
                if idx in indices:
                    continue
                break
            self.prestates[len(indices)] = self.retrieve(idx - 1)
            self.poststates[len(indices)] = self.retrieve(idx)
            indices.append(idx)
        acts = self.actions[indices]
        rews = self.rewards[indices]
        finals = self.finals[indices]
        return self.prestates, acts, rews, self.poststates, finals

    def retrieve(self, idx):
        if idx >= self.history_size - 1:
            return self.observations[idx - self.history_size + 1:idx + 1]
        else:
            indices = [(idx - i) % self.cnt
                       for i in reversed(range(self.history_size))]
            return self.observations[indices]

    def reset(self):
        self.memory.clear()
        self.idx = 0

    def push(self, transition):
        action, reward, observation, done = transition
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.finals[self.idx] = 1 if done else 0
        self.cnt = max(self.cnt, self.idx + 1)
        self.idx = (self.idx + 1) % self.memory_size

    def __len__(self):
        return min(self.cnt, self.memory_size)


class History(object):
    def __init__(self, history_size):
        self.history_size = history_size
        self.reset()

    def push(self, x):
        self.history[1:] = self.history[:-1]
        self.history[0] = x

    def reset(self):
        self.history = np.zeros((4, 84, 84), dtype=np.uint8)

    def get_history(self):
        return self.history.astype(np.float32)


class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, output_size)

        # initializer
        init.xavier_uniform(self.conv1.weight)
        init.constant(self.conv1.bias, 0.1)
        init.xavier_uniform(self.conv2.weight)
        init.constant(self.conv2.bias, 0.1)
        init.xavier_uniform(self.conv3.weight)
        init.constant(self.conv3.bias, 0.1)
        init.xavier_uniform(self.fc1.weight)
        init.constant(self.fc1.bias, 0.1)
        init.xavier_uniform(self.fc2.weight)
        init.constant(self.fc2.bias, 0.1)

    def forward(self, x):
        x /= 255.0
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
        self.memory = ReplayMemory(args.memory_size,
                                   args.history_size,
                                   args.batch_size)
        self.history = History(args.history_size)

        self.steps = 0
        self.train = True

        print('Building DQN')
        self.dqn = DQN(output_size)
        self.target_dqn = DQN(output_size)
        self.criterion = nn.SmoothL1Loss()
        if args.use_cuda:
            self.dqn = self.dqn.cuda()
            self.target_dqn = self.target_dqn.cuda()
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=args.lr,
                                       alpha=args.alpha,
                                       eps=args.rmsprop_ep)

    def q_learn_mini_batch(self):
        args = self.args
        obs, acts, rews, next_obs, finals = self.memory.sample()
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
        loss = self.criterion(prediction_rewards, target_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        q = prediction_rewards.data.cpu().mean()
        return q, loss.data.cpu()[0], True

    def perceive(self, observation, action, reward, done):
        self.history.push(observation)
        result = (0, 0, False)
        if self.train:
            self.memory.push((action, reward, observation, done))
            if self.steps > self.args.learn_start:
                result = self.q_learn_mini_batch()
                if self.steps % self.args.update_freq == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())
        return result

    def qvalue(self, x, use_target=False):
        return self.target_dqn(x) if use_target else self.dqn(x)

    def predict(self, history):
        steps = self.steps
        ep = max(0.1, self.args.epsilon * (1 - steps / 1e6))

        if steps > self.args.learn_start and random() >= ep:
            history = torch.from_numpy(history).view(1, 4, 84, 84).cuda()
            var_his = Variable(history, volatile=True)
            action = self.qvalue(var_his).max(1)[1].cpu().data[0]
        else:
            action = self.env.action_space.sample()
        self.steps += 1
        return action

    def save(self, episodes):
        torch.save(self.dqn.state_dict(),
                   'saves/DQN_{}_{}.bin'.format(self.args.env_name, episodes))


class Statistic(object):
    def __init__(self, output, args):
        self.output = output
        self.args = args
        self.reset()

    def reset(self):
        self.num_game = 0
        self.update_count = 0
        self.ep_reward = 0
        self.total_loss = 0
        self.total_reward = 0
        self.total_q = 0
        self.ep_rewards = []

    def on_step(self, step, reward, done, q, loss, is_update):
        args = self.args
        if step < args.learn_start:
            return

        self.total_q += q
        self.total_loss += loss
        self.total_reward += reward

        if done:
            self.num_game += 1
            self.ep_rewards.append(self.ep_reward)
            self.ep_reward = 0
        else:
            self.ep_reward += reward

        if is_update:
            self.update_count += 1

        update_freq = self.args.update_freq
        if self.update_count % update_freq == 0:
            avg_q = self.total_q / (1e-8 + self.update_count)
            avg_loss = self.total_loss / (1e-8 + self.update_count)
            avg_reward = self.total_reward / args.update_freq
            try:
                max_ep_reward = np.max(self.ep_rewards)
                min_ep_reward = np.min(self.ep_rewards)
                avg_ep_reward = np.mean(self.ep_rewards)
            except Exception as e:
                max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

            self.output({
                'STEP': str(step),
                'AVG_Q': '{:.4f}'.format(avg_q),
                'AVG_L': '{:.4f}'.format(avg_loss),
                'AVG_R': '{:.4f}'.format(avg_reward),
                'EP_MAX_R': '{:.4f}'.format(max_ep_reward),
                'EP_MIN_R': '{:.4f}'.format(min_ep_reward),
                'EP_AVG_R': '{:.4f}'.format(avg_ep_reward),
                'NUM_GAME': '{:.4f}'.format(self.num_game),
            })

            self.reset()
