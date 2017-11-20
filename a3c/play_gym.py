#! /usr/bin/env python3
import argparse
import gym
import roboschool
import scipy.signal

import numpy as np
import tensorflow as tf
import utils as U

from tabulate import tabulate
from discrete_a2c import DiscreteA2C

parser = argparse.ArgumentParser(description='discrete advantage actor critic algorithm')

parser.add_argument(
    '--lr', default=7e-4, type=float, help='learning rate')

parser.add_argument(
    '--max_grad_norm', default=0.5, type=float, help='max gradients normalize')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--seed', default=0, type=int, help='RNG seed')

parser.add_argument(
    '--num_steps', default=5, type=int, help='the number of steps')

parser.add_argument(
    '--num_procs', default=16, type=int, help='the number of processes')

parser.add_argument(
    '--max_steps', default=80e6, type=int, help='max steps of training')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--huber', default=True, type=bool, help='whether to use huber loss')

parser.add_argument(
    '--save_network', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load_network', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--test_alg', default=False, type=bool, help='whether to test our algorithm')

parser.add_argument(
    '--gym_id', default='CartPole-v1', type=str, help='gym id')

parser.add_argument(
    '--model_name', default='discrete_a2c', type=str, help='save or load model name')

args = parser.parse_args()

def build_multi_envs():
    def make_env(rank):
        def _thunk():
            env = gym.make(args.gym_id)
            env.seed(args.seed+rank)
            return env
        return _thunk

    U.set_global_seeds(args.seed)
    env = U.SubproVecEnv([make_env(i) for i in range(args.num_procs)])
    return env

class PlayGym(object):
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent

    def play(self):
        pass
        

    def sample_trajs(self):
        obses, acts, rews, values, dones = [], [], [], [], []
        obs = self.env.reset()
        for step in range(self.args.num_steps):
            obses.append(obs)
            act, value = self.agent.step(obs)
            obs, rew, done, _ = self.env.step(act)
            acts.append(act)
            rews.append(rew)
            values.append(value)
            dones.append(done)



if __name__ == '__main__':
    graph = tf.get_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True,
        intra_op_parallelism_threads=args.num_procs,
        inter_op_parallelism_threads=args.num_procs)

    session = tf.Session(graph=graph, config=config)

    # build env
    env = build_multi_envs()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # build agent
    agent = DiscreteA2C(session, args, obs_dim, act_dim)

    # build player
    player = PlayGym(args, env, agent)

    # start to play :)
    session.run(tf.global_variables_initializer())
    player.play()


