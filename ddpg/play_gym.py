#! /usr/bin/env python3
import argparse
import gym
import roboschool

import numpy as np
import tensorflow as tf

from pddpg import PrioritizedDDPG

parser = argparse.ArgumentParser(description='prioritized deep deterministic policy gradient algorithm')

parser.add_argument(
    '--actor_lr', default=1e-4, type=float, help='actor learning rate')

parser.add_argument(
    '--critic_lr', default=1e-3, type=float, help='critic learning rate')

parser.add_argument(
    '--beta', default=0.4, type=float, help='prioritized replay buffer hyperparameter')

parser.add_argument(
    '--alpha', default=0.6, type=float, help='prioritized replay buffer hyperparameter')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--batch_size', default=64, type=int, help='training batch size')

parser.add_argument(
    '--update_target_num', default=500, type=int, help='the frequence of updating target network')

parser.add_argument(
    '--obs_num', default=1000, type=int, help='how many transitions before agent training')

parser.add_argument(
    '--buffer_size', default=100000, type=int, help='the size of replay buffer')

parser.add_argument(
    '--max_steps', default=100000, type=int, help='max steps of training')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--prioritized', default=True, type=bool, help='whether to use prioritized replay buffer')

parser.add_argument(
    '--huber', default=True, type=bool, help='whether to use huber loss')

parser.add_argument(
    '--save_network', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load_network', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--test_alg', default=False, type=bool, help='whether to test our algorithm')

parser.add_argument(
    '--gym_id', default='RoboschoolInvertedPendulum-v1', type=str, help='gym id')

parser.add_argument(
    '--model_name', default='pddpg', type=str, help='save or load model name')

args = parser.parse_args()

class PlayGym(object):
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent
        self.act_high = env.action_space.high

    def play(self, times=100000):
        for e in range(times):
            score = self._train_episode()
            self.agent.get_score(score)

            if e % 50 == 0:
                scores = [self._test_episode() for _ in range(10)]
                scores = np.asarray(scores)
                print ("Episode: {} | score: {}".format(e, np.mean(scores)))

            if self.agent.time_step > self.args.max_steps:
                break

    def _test_episode(self):
        obs = self.env.reset()
        self.agent.ou_noise.reset()
        done = False
        score = 0
        while not done:
            act = self.agent.action(obs, test=True)
            obs, rew, done, info = self.env.step(self._convert_act(act))
            score += rew

        return score

    def _train_episode(self):
        obs = self.env.reset()
        self.agent.ou_noise.reset()
        done = False
        score = 0
        while not done:
            act = self.agent.action(obs)
            new_obs, rew, done, info = self.env.step(self._convert_act(act))
            self.agent.buffer.add(obs, act, rew, new_obs, done)
            score += rew    
            obs = new_obs
            if len(self.agent.buffer) > self.args.obs_num:
                self.agent.train()

        return score

    def _convert_act(self, act):
        return act * self.act_high


if __name__ == '__main__':
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    env = gym.make(args.gym_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PrioritizedDDPG(session, args, obs_dim, act_dim)
    player = PlayGym(args, env, agent)

    session.run(tf.global_variables_initializer())

    player.play()
