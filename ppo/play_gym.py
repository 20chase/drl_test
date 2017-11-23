#! /usr/bin/env python3
import argparse
import gym
import roboschool

import numpy as np
import tensorflow as tf
import utils as U

from ppo_cliped import PPOCliped

parser = argparse.ArgumentParser(description='proximal policy optimization cliped version')

parser.add_argument(
    '--lr', default=3e-4, type=float, help='learning rate')

parser.add_argument(
    '--ent_coef', default=0., type=float, help='the coefficient of entropy')

parser.add_argument(
    '--clip_range', default=0.2, type=float, help='the clip range parameter')

parser.add_argument(
    '--vf_coef', default=0.5, type=float, help='the coefficient of value function')

parser.add_argument(
    '--max_grad_norm', default=0.5, type=float, help='max gradients normalize')

parser.add_argument(
    '--lamb', default=.95, type=float, help='GAE hyper parameters')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--seed', default=0, type=int, help='RNG seed')

parser.add_argument(
    '--num_batchs', default=4, type=int, help='the number of batchs')

parser.add_argument(
    '--num_opts', default=4, type=int, help='the number of opts')

parser.add_argument(
    '--num_steps', default=2048, type=int, help='the number of steps')

parser.add_argument(
    '--num_procs', default=32, type=int, help='the number of processes')

parser.add_argument(
    '--max_steps', default=10e6, type=int, help='max steps of training')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--save_network', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load_network', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--test_alg', default=False, type=bool, help='whether to test our algorithm')

parser.add_argument(
    '--gym_id', default='RoboschoolAnt-v1', type=str, help='gym id')

parser.add_argument(
    '--model_name', default='ppo_cliped', type=str, help='save or load model name')

args = parser.parse_args()


def make_env():
    env = gym.make(args.gym_id)
    env = U.Monitor(env, '../ppo/{}'.format(args.model_name))
    return env

class PlayGym(object):
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent

    def play(self, max_iters=10000000):
        obs, done = self._reset()
        for i in range(max_iters):
            traj, obs, done = self._sample_traj(obs, done)
            self.agent.learn(traj)
            if i % 500 == 0:
                score = self.test()
                print ("iter: {} | score: {}".format(i, score))
                self.agent.score = score
                if self.args.save_network:
                    self.agent.save_network(self.args.model_name)
                obs, done = self._reset()

    def _reset(self):
        obs = self.env.reset()
        done = False
        return obs, done

    def _sample_traj(self, obs, done):
        obses, acts, rews, values, logps, dones = [], [], [], [], [], []
        for _ in range(self.args.num_steps):
            obs = np.squeeze(obs)
            done = np.squeeze(done)
            obses.append(obs)
            dones.append(done)
            act, value, logp = self.agent.step([obs])
            obs, rew, done, _ = self.env.step(act)
            act = np.squeeze(act)
            rew = np.squeeze(rew)
            value = np.squeeze(value)
            logp = np.squeeze(logp)

            acts.append(act)
            rews.append(rew)
            values.append(value)
            logps.append(logp)

        obs = np.squeeze(obs)
        done = np.squeeze(done)

        obses = np.asarray(obses, dtype=np.float32)
        acts = np.asarray(acts, dtype=np.float32)
        rews = np.asarray(rews, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        logps = np.asarray(logps, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool)

        last_value = self.agent.get_value([obs])[0]

        rets = np.zeros_like(rews)
        advs = np.zeros_like(rews)
        last_gae_lam = 0
        for t in reversed(range(self.args.num_steps)):
            if t == (self.args.num_steps - 1):
                next_no_done = 1. - done
                next_value = last_value
            else:
                next_no_done = 1. - dones[t+1]
                next_value = values[t+1]

            delta = rews[t] + self.args.gamma * next_value * next_no_done - values[t]
            advs[t] = last_gae_lam = delta + self.args.gamma * self.args.lamb * next_no_done * last_gae_lam 

        rets = advs + values
        return [obses, acts, rets, values, logps], obs, done

    def test(self):
        obs = self.env.reset()
        score = 0
        done = False
        while not done:
            act = self.agent.get_action(obs)
            obs, rew, done, _ = self.env.step(act)
            score += np.squeeze(rew)
        return score


if __name__ == '__main__':
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    env = gym.make(args.gym_id)
    ob_space = env.observation_space
    ac_space = env.action_space

    env = U.DummyVecEnv([make_env])
    env = U.VecNormalize(env)

    agent = PPOCliped(session, args, ob_space, ac_space)

    player = PlayGym(args, env, agent)

    session.run(tf.global_variables_initializer())
    player.play()