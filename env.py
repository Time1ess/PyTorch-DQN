from random import randint
import gym


class Env(object):
    def __init__(self, args):
        self._env = gym.make(args.env_name)
        self.skipping_frames = args.skipping_frames
        self.random_start = args.random_start
        self.done = False

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state(self):
        return self.ob, self.reward, self.done, self.info

    def _step(self, action):
        self.ob, self.reward, self.done, self.info = self._env.step(action)

    def step(self, action):
        cumsum = 0
        for skipped in range(self.skipping_frames):
            self._step(action)
            cumsum += self.reward

            if self.done:
                break

        self.reward = cumsum
        return self.state

    def _reset(self):
        self.ob = self._env.reset()
        self.reward = 0
        self.done = False
        self.info = {}

    def reset(self, random=True):
        self.done = False
        self._reset()
        if random:
            for i in range(randint(0, self.random_start - 1)):
                self._step(0)
        return self.ob
