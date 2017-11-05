#! /usr/bin/env python3
import ray
import gym

import numpy as np
import tensorflow as tf
import tensorlayer as tl

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
        envs = [self.make_env.remote() for _ in range(self.num_workers)]
        self.envs = ray.get(envs)

    def reset(self):
        states = [self.env_reset.remote(self.envs[i]) for i in range(self.num_workers)]
        states = ray.get(states)
        return states

    def step(self, states):
        next_step = [self.env_step.remote(states[i]) for i in range(self.num_workers)]
        next_step = ray.get(next_step)
        
        states = [batch[0] for batch in next_step]
        rewards = [batch[1] for batch in next_step]
        dones = [batch[2] for batch in next_step]

        return [states, rewards, dones]

    @ray.remote
    def make_env(self):
        return gym.make(self.env_name)

    @ray.remote
    def env_reset(self, env):
        return env.reset()

    @ray.remote
    def env_step(self, state):
        return env.step(state)     

def run_episode(env, agent):
    dones = [False for _ in range(env.num_workers)]
    dones_idxs = [0 for _ in range(env.num_workers)]

    obs, acts, rews = [], [], []

    states = env.reset()
    while not all(dones):
        obs.append(states)
        actions = agent.take_action(states)
        acts.append(actions)

        
        

        rewards.append(reward)

    obs = np.asarray(obs)
    acts = np.asarray(acts)
    rewards = np.asarray(rewards)

    trajectory = {
    'obs': obs,
    'acts': acts,
    'rewards': rewards
    }

    return total_rewards

if __name__ == '__main__':
    ray.init()
    env = 'Pendulum-v0'
    agent = Agent(env)
    ray_env = RayEnvironment(env, 4)
    trajectories = run_episode(ray_env, agent)
    print (trajectories)




