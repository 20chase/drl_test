#! /usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='prioritized dueling double deep q-network algorithm')

parser.add_argument(
	'--lr', default=1e-4, type=float, help='learning rate')

parser.add_argument(
	'--final_epsilon', default=0.2, type=float, help='epsilon greedy exploration hyperparameter')

parser.add_argument(
	'--beta', default=0.4, type=float, help='prioritized replay buffer hyperparameter')

parser.add_argument(
	'--alpha', default=0.6, type=float, help='prioritized replay buffer hyperparameter')

parser.add_argument(
    '--gamma', default=0.99, type=float, help='gamma')

parser.add_argument(
	'--batch_size', default=32, type=int, help='training batch size')

parser.add_argument(
	'--update_target_num', default=500, type=int, help='the frequence of updating target network')

parser.add_argument(
	'--obs_num', default=1000, type=int, help='how many transitions before agent training')

parser.add_argument(
	'--explore_num', default=50000, type=int, help='how many transitions finished the exploration')

parser.add_argument(
	'--buffer_size', default=100000, type=int, help='the size of replay buffer')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--save_network', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load_network', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--gym_id', default='CartPole-v0', type=str, help='gym id')

parser.add_argument(
    '--model_name', default='pddqn', type=str, help='save or load model name')