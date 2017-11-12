#! /usr/bin/env python3
import ray
import gym
import time
import roboschool

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from collections import defaultdict

class Agent(object):
    def __init__(self, env):
        self.init_param(env)
        self.sess = tf.InteractiveSession()

        self._build_ph()
        self.network, self.action = self._build_network()

        self.sess.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')

    def _build_network(self):
        network = tl.layers.InputLayer(self.obs_ph, name='input')
        network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.relu, name='hide1')
        network = tl.layers.DenseLayer(network, n_units=self.act_dim, act=tf.nn.tanh, name='out')
        outputs = network.outputs
        action = outputs * self.act_high
        return network, action

    def take_action(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.action, feed_dict)

    def init_param(self, env):
        env = gym.make(env)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_high = env.action_space.high

@ray.remote
class RayEnvironment(object):
    def __init__(self, env):
        self.env = env
        state = self.env.reset()
        self.shape = state.shape

    def step(self, action):
        if self.done:
            return [np.zeros(self.shape), 0.0, True]
        else:
            state, reward, done, info = self.env.step(action)
            self.done = done
            return [state, reward, done]

    def reset(self):
        self.done = False
        return self.env.reset()
        
    
def run_episode(envs, agent):
    terminates = [False for _ in range(len(envs))]
    terminates_idxs = [0 for _ in range(len(envs))]

    paths = defaultdict(list)

    states = [env.reset.remote() for env in envs]
    states = ray.get(states)
    # print (states)
    while not all(terminates):
        paths["states"].append(states)

        actions = agent.take_action(states)
        paths["actions"].append(actions)

        next_step = [env.step.remote(actions[i]) for i, env in enumerate(envs)]
        next_step = ray.get(next_step)

        # print (next_step)

        states = [batch[0] for batch in next_step]
        rewards = [batch[1] for batch in next_step]
        dones = [batch[2] for batch in next_step]

        paths["rewards"].append(rewards)

        for i, d in enumerate(dones):
            if d:
                terminates[i] = True
            else:
                terminates_idxs[i] += 1

    return paths

if __name__ == '__main__':
    ray.init()
    env_name = 'Pendulum-v0'
    # env_name = 'RoboschoolAnt-v1'
    envs = [gym.make(env_name) for _ in range(8)]
    agent = Agent(env_name)
    envs = [RayEnvironment.remote(envs[i]) for i in range(2)]

    for i in range(100000):
        trajectories = run_episode(envs, agent)
        print (i)





