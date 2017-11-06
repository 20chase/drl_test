#! /usr/bin/env python3
import ray
import gym

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


class RayEnvironment(object):
    def __init__(self, env_name, num_workers):
        self.env_name = env_name
        self.num_workers = num_workers

        self.build_envs()
        
    def build_envs(self):
        self.envs = [gym.make(self.env_name) for _ in range(self.num_workers)]
        
    def reset(self):
        @ray.remote
        def env_reset(env):
            return env.reset()

        states = [env_reset.remote(self.envs[i]) for i in range(self.num_workers)]
        states = ray.get(states)
        return states

    def step(self, states):
        @ray.remote
        def env_step(env, state):
            return env.step(state) 

        next_step = [env_step.remote(self.envs[i], states[i]) for i in range(self.num_workers)]
        next_step = ray.get(next_step)

        states = [batch[0] for batch in next_step]
        rewards = [batch[1] for batch in next_step]
        dones = [batch[2] for batch in next_step]

        return [states, rewards, dones]

    

def run_episode(env, agent):
    terminates = [False for _ in range(env.num_workers)]
    terminates_idxs = [0 for _ in range(env.num_workers)]

    paths = defaultdict(list)

    states = env.reset()
    while not all(terminates):
        paths["states"].append(states)

        actions = agent.take_action(states)
        paths["actions"].append(actions)

        states, rewards, dones = env.step(actions)
        paths["rewards"].append(rewards)

        for i, d in enumerate(dones):
            if d:
                terminates[i] = True
            else:
                terminates_idxs[i] += 1

    return paths

if __name__ == '__main__':
    ray.init()
    env = 'Pendulum-v0'
    agent = Agent(env)
    ray_env = RayEnvironment(env, 2)

    # print ('end')
    trajectories = run_episode(ray_env, agent)
    # print (trajectories)




