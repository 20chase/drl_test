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

		self.network, self.action = self.self._build_ph()

		self.session.run(tf.global_variables_initializer())

	def _build_ph(self):
		self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')

	def _build_network(self):
		network = tl.layers.InputLayer(self.obs_ph, name='input')
		network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.relu, name='hide1')
		network = tl.layers.DenseLayer(network, n_units=self.act_dim, act=tf.nn.tanh, name='out')
		output = network.outputs
		action = outputs * self.act_high
		return network, action

	def take_action(self, obs):
		obs = np.reshape(obs, (1, self.obs_dim))
        feed_dict = {self.obs_ph: obs}
		return self.sess.run(self.action, feed_dict)

	def init_param(self, env):
		self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_high = env.action_space.high


