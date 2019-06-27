import argparse
import gym
import time
import logger
import os, sys

import numpy as np
import tensorflow as tf
import utils as U
import monitor as M

from td3 import TD3
from collections import deque

parser = argparse.ArgumentParser(description='prioritized deep deterministic policy gradient algorithm')

parser.add_argument(
    '--train', action='store_true')

parser.add_argument(
    '--actor_lr', default=1e-3, type=float, help='actor learning rate')

parser.add_argument(
    '--critic_lr', default=1e-3, type=float, help='critic learning rate')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--sigma', default=0.1, type=float)

parser.add_argument(
    '--nenvs', default=16, type=int, help='the number of processes')

parser.add_argument(
    '--batch_size', default=100, type=int, help='training batch size')

parser.add_argument(
    '--replay_size', default=1000000, type=int, help='the size of replay buffer')

parser.add_argument(
    '--max_steps', default=500000, type=int, help='max steps of training')

parser.add_argument(
    '--huber', default=True, type=bool, help='whether to use huber loss')

parser.add_argument(
    '--save', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load', action='store_true')

parser.add_argument(
    '--gym_id', default='Humanoid-v3', type=str, help='gym id')

parser.add_argument(
    '--seed', default=0, type=int)

args = parser.parse_args()

class PlayGym(object):
    def __init__(self, 
                 args, 
                 train_env, test_env, 
                 agent):

        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent

        self.nsteps = 200

    def learn(self, 
              start_steps=100000, 
              steps_per_epoch=10000, 
              epochs=5000,
              max_ep_len=1000):

        ob = self.train_env.reset()
        ep_len = 0
        ep_ret = 0

        total_steps = steps_per_epoch * epochs + 1
        for t in range(1, total_steps):
            if t > start_steps:
                act = self.agent.action([ob])[0]
            else:
                act = self.train_env.action_space.sample()

            new_ob, rew, done, _ = self.train_env.step(act)
            ep_len += 1
            ep_ret += rew

            done = False if ep_len==max_ep_len else done

            self.agent.replay_buffer.store(ob, act, rew, new_ob, done)

            ob = new_ob

            if done or (ep_len == max_ep_len):
                self.agent.train(ep_len)

                print("time_step {}: {}".format(t, ep_ret))

                ob = self.train_env.reset()
                ep_len = 0
                ep_ret = 0

            if t % 10000 == 0:
                self.agent.save_net("./log/{}".format(int(t / 10000)))

    def play(self):
        ob = self.test_env.reset()
        done = False
        total_rew = 0
        while not done:
            act = self.agent.action([ob], test=True)[0]
            self.test_env.render()
            ob, rew, done, _ = self.test_env.step(act)
            total_rew += rew
        
        print("reward: {}".format(total_rew))


class MakeEnv(object):
    def __init__(self):
        pass

    def make(self):
        return gym.make(args.gym_id)



if __name__ == '__main__':
    curr_path = sys.path[0]
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    maker = MakeEnv() 
    train_env, test_env = maker.make(), maker.make()

    ob_space = train_env.observation_space
    ac_space = train_env.action_space
    ac_high = ac_space.high[0]

    agent = TD3(session, args, 
        ob_space.shape[0], ac_space.shape[0])
    player = PlayGym(
        args, train_env, test_env, agent
        )

    session.run(tf.global_variables_initializer())
    session.run(agent.target_init)
    if args.load:
        agent.load_net("./log/4")
    if args.train:
        player.learn()
    else:
        player.play()
