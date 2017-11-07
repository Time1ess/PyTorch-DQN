from random import randrange

import numpy as np
from scipy.misc import imresize


def preprocess(rgb):
    """
    rgb: height x width x channel(RGB)
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    x = 0.2126 * r + 0.7152 * g + 0.0722 * b
    x = imresize(x, (84, 84)).astype(np.uint8)
    return x


def random_start(env, max_random_frame):
    env.reset()
    frame = randrange(max_random_frame)
    legal_actions = env.action_space.n
    for _ in range(frame):
        env.step(randrange(legal_actions))
    return env.env._get_obs()
