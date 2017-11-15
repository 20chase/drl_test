import tensorflow as tf
import tensorlayer as tl
import gym
import threading

import numpy as np
import random
import os
from gym import wrappers
from collections import deque

# hyper parameter

ENV_NAME = 'CartPole-v1'

VALUE_LR = 2.5e-4
POLICY_LR = 2.5e-4
GAMMA_NUM = 0.99
THREAD_NUM = 8
REPLAY_SIZE = 200000 
BATCH_SIZE = 128
OBSERVE_NUM = 100
TIME_MAX = 512


episode = 0


class A3CAgent():

	def __init__(self, env_name):

		self.init_param(env_name)

		self.build_value_network()
		self.build_policy_network()
		self.creat_training_method()
		
		self.session = tf.InteractiveSession()
		self.merge_all = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter('/tmp/train', self.session.graph)
		self.session.run(tf.global_variables_initializer()) 

	def build_value_network(self):

		self.value_input = tf.placeholder("float", [None, self.state_dim])

		self.value_network = tl.layers.InputLayer(self.value_input, name = 'value_input')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 128, act = tf.nn.relu, name = 'value_relu1')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 128, act = tf.nn.relu, name = 'value_relu2')
		# self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 256, act = tf.nn.relu, name = 'value_relu3')
		# self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 256, act = tf.nn.relu, name = 'value_relu4')
		# self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 128, act = tf.nn.relu, name = 'value_relu5')
		# self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 128, act = tf.nn.relu, name = 'value_relu6')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 1, name = 'value_output')

		self.value = self.value_network.outputs

	def build_policy_network(self):

		self.policy_input = tf.placeholder("float", [None, self.state_dim])

		self.policy_network = tl.layers.InputLayer(self.policy_input, name = 'policy_input')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 128, act = tf.nn.relu, name = 'policy_relu1')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 128, act = tf.nn.relu, name = 'policy_relu2')
		# self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 256, act = tf.nn.relu, name = 'policy_relu3')
		# self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 256, act = tf.nn.relu, name = 'policy_relu4')
		# self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 128, act = tf.nn.relu, name = 'policy_relu5')
		# self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 128, act = tf.nn.relu, name = 'policy_relu6')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = self.action_dim, name = 'policy_output')

		self.policy_logits = self.policy_network.outputs

		self.policy = tf.nn.softmax(self.policy_logits)

	def creat_training_method(self):

		self.action_input = tf.placeholder("float", [None, self.action_dim])
		self.target_input = tf.placeholder("float", [None])
		# self.current_value = tf.placeholder("float", [None, self.action_dim])

		self.score_input = tf.placeholder("float")
		self.reward_sum = tf.placeholder("float")
		self.replay_size = tf.placeholder("float")

		policy = tf.nn.softmax_cross_entropy_with_logits(logits = self.policy_logits, labels = self.action_input)
		advantage = self.target_input - self.value
		entropy = tf.reduce_sum(self.policy * tf.log(self.policy), axis = 1)

		value_cost = tf.reduce_mean(tf.square(self.target_input - self.value))
		policy_cost = tf.reduce_sum(tf.multiply(policy, tf.stop_gradient(advantage)))

		value_summary = tf.reduce_mean(self.value)

		with tf.name_scope('cost'):
			tf.summary.scalar('value_cost', value_cost)

		with tf.name_scope('value'):
			tf.summary.scalar('score', self.score_input)
			tf.summary.scalar('value', value_summary)


		self.value_opt = tf.train.AdamOptimizer(VALUE_LR).minimize(value_cost)
		self.policy_opt = tf.train.AdamOptimizer(POLICY_LR).minimize(policy_cost)

	def train_network(self, state_batch, action_batch, reward_batch, next_state, done):

		self.time_step += 1

		reward_batch = self.computer_reward(next_state, reward_batch, done)

		state_batch = np.asarray(state_batch)
		action_batch = np.asarray(action_batch)
		reward_batch = np.asarray(reward_batch)
	
	
		feed_dict = {
		self.value_input:state_batch,
		self.policy_input:state_batch,
		self.target_input:reward_batch,
		self.action_input:action_batch,
		self.score_input:self.score
		}

		summary, _, _ = self.session.run([self.merge_all, self.value_opt, self.policy_opt], feed_dict = feed_dict)

		self.writer.add_summary(summary, self.time_step)


	def init_param(self, env_name):

		self.env_name = env_name

		self.replay_buffer = deque()

		env = gym.make(env_name)

		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n

		env.close()

		self.time_step = 0
		self.score = 0

	def computer_reward(self, next_state, origin_reward, done):
		origin_reward = np.asarray(origin_reward)
		reward_batch = np.zeros_like(origin_reward, dtype = np.float32)

		discounted_item = 0

		next_state = next_state.reshape(1, self.state_dim)

		if not done:
			discounted_item = self.get_value(next_state)

		for i in reversed(xrange(len(origin_reward))):
			
			discounted_item = discounted_item * GAMMA_NUM + origin_reward[i]
			reward_batch[i] = discounted_item

		return reward_batch 

	def get_value(self, state):
		return self.value.eval(feed_dict = {self.value_input: state}, session = self.session)

	def get_policy(self, state):
		return self.policy.eval(feed_dict = {self.policy_input: state}, session = self.session)

	def get_action(self, state):

		state = state.reshape(1, self.state_dim)

		action_prob = self.get_policy(state)

		return np.random.choice(range(self.action_dim), p=action_prob.flatten())

	def one_hot_process(self, action):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		return one_hot_action

	def write_score(self, score):
		self.score = score


class Agent(threading.Thread):

	def __init__(self, index, global_agent):

		threading.Thread.__init__(self)

		self.state_batch = []
		self.reward_batch = []
		self.action_batch = []

		self.agent = global_agent


	def run(self):
		env = gym.make(self.agent.env_name)

		global episode

		while True:

			state = env.reset()
			score = 0

			self.state_batch = []
			self.action_batch = []
			self.reward_batch = []

			time_step = 0

			while True:

				action = self.agent.get_action(state)
				next_state, reward, done, _ = env.step(action)
				score += reward

				time_step += 1

				self.memory(state, self.agent.one_hot_process(action), reward)

				if time_step > TIME_MAX or done:
					self.agent.train_network(self.state_batch, self.action_batch, self.reward_batch, next_state, done)
					self.state_batch = []
					self.action_batch = []
					self.reward_batch = []
					time_step = 0

				state = next_state

				if done:
					episode += 1
					print 'episode: ', episode, ' | score : ', score
					self.agent.write_score(score)
					break

	def memory(self, state, action, reward):
		self.state_batch.append(state)
		self.action_batch.append(action)
		self.reward_batch.append(reward)

		# self.agent.replay_buffer.append((state,one_hot_action,reward,next_state,done))


if True:

	global_agent = A3CAgent(ENV_NAME)

	agents = [Agent(i, global_agent) for i in range(THREAD_NUM)]

	for agent in agents:
		agent.start()
		




	









