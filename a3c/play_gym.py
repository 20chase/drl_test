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
    '--ent_coef', default=0., type=float, help='the coefficient of entropy')

parser.add_argument(
    '--vf_coef', default=0.5, type=float, help='the coefficient of value function')

parser.add_argument(
    '--max_grad_norm', default=0.5, type=float, help='max gradients normalize')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--seed', default=0, type=int, help='RNG seed')

parser.add_argument(
    '--num_steps', default=5, type=int, help='the number of steps')

parser.add_argument(
    '--num_procs', default=32, type=int, help='the number of processes')

parser.add_argument(
    '--max_steps', default=8e6, type=int, help='max steps of training')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--softmax', default=True, type=bool, help='whether to use softmax to sample action')

parser.add_argument(
    '--huber', default=False, type=bool, help='whether to use huber loss')

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
    env = U.SubprocVecEnv([make_env(i) for i in range(args.num_procs)])
    return env

class PlayGym(object):
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent
        self.test_env = gym.make(self.args.gym_id)

    def play(self, max_iters=100000):
        obs = self.env.reset()
        for i in range(max_iters):
            obses, acts, rews, values, obs = self._sample_trajs(obs)
            self.agent.update(obses, acts, rews, values)
            if i % 100 == 0:
                score = self.test()
                print ("iter: {} | score: {}".format(i, score))
                self.agent.score = score

    def test(self):
        env = self.test_env
        obs = env.reset()
        score = 0
        done = False
        while not done:
            act = self.agent.get_action([obs])
            obs, rew, done, _ = env.step(act)
            score += rew
        return score

    def _sample_trajs(self, obs):
        obses, acts, rews, values, dones = [], [], [], [], []
        
        for step in range(self.args.num_steps):
            obses.append(obs)
            act, value = self.agent.step(obs)
            obs, rew, done, _ = self.env.step(act)
            acts.append(act)
            rews.append(rew)
            values.append(value)
            dones.append(done)

        obses = np.asarray(obses, dtype=np.float32).swapaxes(1, 0)
        acts = np.asarray(acts, dtype=np.int32).swapaxes(1, 0)
        rews = np.asarray(rews, dtype=np.float32).swapaxes(1, 0)
        values = np.asarray(values, dtype=np.float32).swapaxes(1, 0)
        dones = np.asarray(dones, dtype=np.bool).swapaxes(1, 0)
        last_values = self.agent.get_value(obs)

        for n, (rew, done, value) in enumerate(zip(rews, dones, last_values)):
            rew = rew.tolist()
            done = done.tolist()
            if done[-1] == 0:
                rew = U.discount_with_dones(rew+[value], done+[0.], self.args.gamma)[:-1]
            else:
                rew = U.discount_with_dones(rew, done, self.args.gamma)
            rews[n] = rew

        obses = np.concatenate([obs for obs in obses])
        acts = acts.flatten()
        rews = rews.flatten()
        values = values.flatten()

        return obses, acts, rews, values, obs 

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


