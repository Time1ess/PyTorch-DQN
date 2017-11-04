import argparse

import torch


parser = argparse.ArgumentParser(description='Hyper parameters to train DQN')
parser.add_argument('-N', '--memory_size', type=int, default=300000,
                    help='Size of replay memory(default: 300k)')
parser.add_argument('--random_seed', type=int, default=1234,
                    help='Random state seed(default: 1234)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size of replay memory(default: 32)')
parser.add_argument('--max_episodes', type=int, default=50000000,
                    help='Max episodes to train DQN(default: 10m)')
parser.add_argument('--epsilon', type=float, default=1,
                    help='Init epsilon for randomizing actions(default: 1)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Decay rate for later rewards(default: 0.99)')
parser.add_argument('--history_size', type=int, default=4,
                    help='Number of latest frames to preprocess(default: 4)')
parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available(),
                    help='Use CUDA to accelerate if available(default: True)')
parser.add_argument('--max_episodes_per_epoch', type=int, default=100000,
                    help='Number of episodes in each epoch(default: 100000)')
parser.add_argument('--fixed_history_size', type=int, default=1000,
                    help='Number of evaluation set(default: 1000)')
parser.add_argument('--skipping_frames', type=int, default=4,
                    help='Number of skipping frames(default: 4)')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='Learning rate(default: 0.00025)')
parser.add_argument('--alpha', type=float, default=0.95,
                    help='Alpha for RMSprop(default: 0.95)')
parser.add_argument('--rmsprop_ep', type=float, default=0.01,
                    help='Epsilon for RMSprop(default: 0.01)')
parser.add_argument('--random_start', type=float, default=30,
                    help='Play frames in random action when start(default: 30)')
parser.add_argument('--eval_freq', type=float, default=250000,
                    help='Evaluation frequency(default: 250000)')
parser.add_argument('--eval_steps', type=float, default=125000,
                    help='Evaluation steps(default: 125000)')
parser.add_argument('--update_freq', type=float, default=10000,
                    help='Update Target DQN frequency(default: 10000)')
parser.add_argument('--save_freq', type=float, default=125000,
                    help='Save frequency(default: 125000)')
parser.add_argument('--learn_start', type=float, default=50000,
                    help=('Number of episodes waitting begin to learn'
                          '(default: 50000)'))
parser.add_argument('--checkpoint', type=str,
                    help='Model check point path')
parser.add_argument('--env_name', type=str, default='Breakout-v0',
                    help='OpenAI gym name')

args = parser.parse_args()

# Tensor types
if args.use_cuda:
    args.ByteTensor = torch.cuda.ByteTensor
    args.FloatTensor = torch.cuda.FloatTensor
    args.LongTensor = torch.cuda.LongTensor
else:
    args.ByteTensor = torch.ByteTensor
    args.FloatTensor = torch.FloatTensor
    args.LongTensor = torch.LongTensor
