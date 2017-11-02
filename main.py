import logging
import os.path as osp

from tqdm import tqdm

from configs import args
from model import DQNAgent
from utils import preprocess, log_print
from env import Env


env = Env(args)
legal_actions = env.action_space
agent = DQNAgent(env, legal_actions.n, args)

logging.basicConfig(
    filename=osp.join('logs', args.env_name + '.log'),
    filemode='w',
    format='%(asctime)s:%(message)s',
    level=logging.DEBUG)

observation = preprocess(env.reset())
agent.reset_history()
agent.push_observation(observation, repeat=args.history_size)
total_progress = tqdm(range(1, 1 + args.max_episodes), miniters=100)
epsum = 0
epcnt = 0
totsum = 0
totcnt = 0
for episodes in total_progress:
    history = agent.get_history()
    action = agent.predict(history)
    observation, reward, done, info = env.step(action)
    observation = preprocess(observation)

    if reward > 0:
        reward = 1
    elif reward < 0:
        reward = -1

    epsum += reward
    epcnt += 1
    epavg = epsum / epcnt
    totsum += reward
    totcnt += 1
    totavg = totsum / totcnt

    agent.perceive(observation, action, reward, done)

    if episodes % args.eval_freq == 0:
        log_print('Update Target DQN')
        total_progress.set_postfix({'EP_AVG': epavg, 'TOT_AVG': totavg})
        totsum = 0
        totcnt = 0
        agent.update_target_dqn()
    if episodes % args.save_freq == 0:
        log_print('Save DQN')
        agent.save(episodes)
    if done:
        observation = preprocess(env.reset())
        agent.reset_history()
        agent.push_observation(observation, repeat=args.history_size)
        total_progress.set_postfix({'EP_AVG': epavg, 'TOT_AVG': totavg})
        epsum = 0
        epcnt = 0
