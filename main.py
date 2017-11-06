import logging
from tqdm import tqdm

from configs import args
from model import DQNAgent, Statistic
from utils import preprocess
from env import Env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='logs/DQN.log',
    filemode='w')


env = Env(args)
legal_actions = env.action_space
agent = DQNAgent(env, legal_actions.n, args)

agent.history.reset()
observation = preprocess(env.reset())
agent.push_observation(observation, repeat=args.history_size)
total_progress = tqdm(range(1, 1 + args.max_episodes), miniters=100)


def logging_stats(info_dict):
    logging.info(' '.join(
        (key + ':' + val for key, val in sorted(info_dict.items()))))
    total_progress.set_postfix(info_dict)


stat = Statistic(logging_stats, args)
for episodes in total_progress:
    history = agent.history.get_history()
    action = agent.predict(history)
    observation, reward, done, info = env.step(action)
    observation = preprocess(observation)

    reward = max(-1, min(1, reward))
    qvalue, loss, is_update = agent.perceive(observation, action, reward, done)

    stat.on_step(episodes, reward, done, qvalue, loss, is_update)

    if episodes % args.save_freq == args.save_freq - 1:
        agent.save(episodes)
    if done:
        agent.history.reset()
        observation = preprocess(env.reset())
        agent.push_observation(observation, repeat=args.history_size)
