# author: fan tingxiang
# data: 2017-5-17
# e-mail: ftx1994@foxmail.com
# theme: policy gradient
#-----------------------------

# dependences
import gym
import time
import os
import random

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from gym import wrappers
from collections import deque

# hyper parameters

LEARNING_RATE = 1e-3

REPLAY_SIZE = 1e4
BATCH_SIZE = 256

GAMMA_NUM = 1
OBSERVATION_NUM = 1
PRINT_NUM = 100
EPISODE_NUM = 100000
STEP_NUM = 10000

SAVE_FLAG = False
LOAD_FLAG = False
DISPLAY_FLAG = False
EVAL_FLAG = False

MODE_NAME = 'LunarLander-v2'
# MODE_NAME = 'CartPole-v0'

class Policy_Gradient():
	def __init__(self, env):

		self.time_step = 0
		self.episode = 0
		self.reward = 0

		self.replay_buffer = deque()
		self.reward_buffer = deque()

		self.action_dim = env.action_space.n
		self.state_dim = env.observation_space.shape[0]

		self.creat_network()

		self.creat_train()

		self.session = tf.InteractiveSession()

		self.merged = tf.summary.merge_all()

		self.train_writer = tf.summary.FileWriter('/tmp/policy', self.session.graph)

		self.session.run(tf.global_variables_initializer())

	def creat_network(self):

		self.state_input = tf.placeholder(tf.float32, shape=[None, self.state_dim])

		self.network = tl.layers.InputLayer(self.state_input)
		self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu1')
		self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu2')
		self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu3')
		self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu4')
		self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu5')
		self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu6')
		self.network = tl.layers.DenseLayer(self.network, n_units=self.action_dim, name='output')

		self.network_output = self.network.outputs

		self.output_prob = tf.nn.softmax(self.network_output)

	def creat_train(self):
		self.action_input = tf.placeholder(tf.float32, shape=[None, self.action_dim])
		self.reward_input = tf.placeholder(tf.float32, shape=[None])

		self.reward_sum = tf.placeholder(tf.float32)
		self.reward_baseline = tf.placeholder(tf.float32)
		self.replay_size = tf.placeholder(tf.float32)

		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.network_output, labels = self.action_input)

		self.cost = tf.reduce_sum(tf.multiply(self.cross_entropy, self.reward_input))

		with tf.name_scope('cost'):
			tf.summary.scalar('cost', self.cost)

		with tf.name_scope('reward'):
			tf.summary.scalar('reward_sum', self.reward_sum)
			tf.summary.scalar('reward_baseline', self.reward_baseline)

		with tf.name_scope('param'):
			tf.summary.scalar('replay_size', self.replay_size)

		self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.cost)
		# self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

	def perceive(self, state_batch, action_batch, reward_batch, terminate_batch):
		self.episode += 1
		state_batch = np.asarray(state_batch)
		action_batch = np.asarray(action_batch)
		reward_batch = self.computer_reward(reward_batch, terminate_batch)
		
		for i in xrange(action_batch.shape[0]):
			self.replay_buffer.append((state_batch[i], action_batch[i], reward_batch[i]))
			self.reward_buffer.append(reward_batch[i])


		while len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()
			self.reward_buffer.popleft()

	def train_network(self):
		self.time_step += 1

		minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]


		reward_buffer = np.asarray(self.reward_buffer)
		reward_baseline = np.mean(reward_buffer)
		reward_batch -= reward_baseline
		reward_batch /= np.std(reward_buffer)

		replay_size = len(self.replay_buffer)

		summary, _ = self.session.run([self.merged, self.optimizer], feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch,
			self.reward_input:reward_batch,
			self.reward_sum:self.reward,
			self.replay_size:replay_size,
			self.reward_baseline:reward_baseline
			})

		self.train_writer.add_summary(summary, self.time_step)
		
	def take_action(self, state):

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_network()

		state = state.reshape(1, self.state_dim)
		action_prob = self.output_prob.eval(feed_dict={self.state_input: state})

		return np.random.choice(range(self.action_dim), p=action_prob.flatten())

	def write_reward(self, reward_sum):
		self.reward = reward_sum

	def computer_reward(self, origin_reward = [], terminate_batch = []):
		origin_reward = np.asarray(origin_reward)
		reward_batch = np.zeros_like(origin_reward, dtype = np.float32)

		for i in reversed(xrange(0, origin_reward.size)):
			if terminate_batch[i]:
				discounted_item = 0

			discounted_item = discounted_item * GAMMA_NUM + origin_reward[i]
			reward_batch[i] = discounted_item

		return reward_batch 

	def one_hot_process(self, action):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		return one_hot_action

	def print_reward(self):
		reward_buffer = np.asarray(self.reward_buffer)
		print 'reward_mean: ', np.mean(reward_buffer)
		



def train_game():

	env = gym.make(MODE_NAME)
	if EVAL_FLAG:
		env = wrappers.Monitor(env, '/tmp/' + MODE_NAME)

	agent = Policy_Gradient(env)

	if LOAD_FLAG:
		network_params = tl.files.load_npz(name = MODE_NAME + '.npz')
		tl.files.assign_params(agent.session, network_params, agent.network)

	reward_mean = 0
	reward_sum = 0
	end_flag = False

	for episode in xrange(EPISODE_NUM):

		state = env.reset()

		state_batch = []
		action_batch = []
		reward_batch = []
		terminate_batch = []

		if end_flag:
			break

		for step in xrange(STEP_NUM):

			if DISPLAY_FLAG:
				env.render()

			action = agent.take_action(state)

			next_state, reward, terminate, _ = env.step(action)

			reward_sum += reward

			

			state_batch.append(state)
			action_batch.append(agent.one_hot_process(action))
			terminate_batch.append(terminate)

			state = next_state

			if terminate:

				reward_batch.append(reward_sum)

				agent.perceive(state_batch, action_batch, reward_batch, terminate_batch)

				agent.write_reward(reward_sum)

				print 'episode: ', episode, '... reward_sum:', reward_sum
				reward_mean += reward_sum
				reward_sum = 0

				agent.print_reward()

				if (episode % PRINT_NUM == 0) and (episode != 0):
					if SAVE_FLAG:
						tl.files.save_npz(agent.network.all_params, name = MODE_NAME + '.npz')

					reward_mean /= (PRINT_NUM + 1)

					if reward_mean > 195:
						end_flag = True

					print 'episode:', episode, '... reward_mean: ', reward_mean

				break

			else:
				reward_batch.append(0)


if __name__ == '__main__':
  	train_game()
  	if EVAL_FLAG:
		gym.upload('/tmp/' + MODE_NAME, api_key='sk_nXYWtyR0CfjmTgSiJVJA')
 






