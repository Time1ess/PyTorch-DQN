import logging
import os.path as osp

from tqdm import tqdm

from configs import args
from model import DQNAgent
from utils import preprocess
from env import Env


env = Env(args)
legal_actions = env.action_space
agent = DQNAgent(env, legal_actions.n, args)

agent.reset_history()
observation = preprocess(env.reset())
agent.push_observation(observation, repeat=args.history_size)
total_progress = tqdm(range(1, 1 + args.max_episodes), miniters=100)
epsum = 0
epcnt = 0
totsum = 0
for episodes in total_progress:
    history = agent.get_history()
    action = agent.predict(history)
    observation, reward, done, info = env.step(action)
    observation = preprocess(observation)

    reward = max(-1, min(1, reward))

    epsum += reward
    epcnt += 1
    epavg = epsum / epcnt
    totsum += reward
    totavg = totsum / args.eval_freq

    agent.perceive(observation, action, reward, done)

    msg = {'EPSILON': '{:.5f}'.format(agent.epsilon),
           'EP_AVG': '{:.6f}'.format(epavg),
           'TOT_AVG': '{:.6f}'.format(totavg),
           'Q_AVG': '{:.6f}'.format(
               agent.total_qvalue / (1e-8 + agent.update_count))}
    total_progress.set_postfix(msg)

    if episodes % args.eval_freq == 0:
        totsum = 0
        agent.total_qvalue = 0
        agent.update_count = 0
    if episodes % args.save_freq == 0:
        agent.save(episodes)
    if done:
        agent.reset_history()
        observation = preprocess(env.reset())
        agent.push_observation(observation, repeat=args.history_size)
        epsum = 0
        epcnt = 0
