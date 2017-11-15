import tensorflow as tf
import tensorlayer as tl
import gym
import time
import filter_env

import numpy as np
import random
import os
from gym import wrappers
from collections import deque

# hyper param

VALUE_LR = 1e-3
POLICY_LR = 1e-4
TAU_NUM = 0.01
GAMMA_NUM = 0.99
MU_NUM = 0
THETA_NUM = 0.15
INIT_SIGMA = 0.2
REPLAY_SIZE = 100000
BATCH_SIZE = 32
TEST = 100

# ENV_NAME = 'InvertedPendulum-v1'
ENV_NAME = 'InvertedDoublePendulum-v1'
# ENV_NAME = 'LunarLanderContinuous-v2'


def tic():
	globals()['tt'] = time.clock()
 
def toc():
	print '\nElapsed time: %.8f seconds\n' % (time.clock()-globals()['tt'])

class DDPG():

	def __init__(self, env):

		self.init_param(env)

		self.build_Qvalue_network()
		self.build_Qvalue_network_target()
		self.build_policy_network()
		self.build_policy_network_target()
		self.creat_training_method()

		self.session = tf.InteractiveSession()
		self.merge_all = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter('/tmp/train', self.session.graph)
		self.session.run(tf.global_variables_initializer()) 

	def build_Qvalue_network(self):

		self.value_input = tf.placeholder("float", [None, self.state_dim], name = 'value_input_ph')
		self.value_network = tl.layers.InputLayer(self.value_input, name = 'value_input')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 400, act = tf.nn.relu, name = 'value_relu1')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 300, name = 'value_output')


		self.action_input = tf.placeholder("float", [None, self.action_dim], name = 'action_input_ph')
		self.action_network = tl.layers.InputLayer(self.action_input, name = 'action_input')
		self.action_network = tl.layers.DenseLayer(self.action_network, n_units = 300, name = 'action_output')

		action_output = self.action_network.outputs
		value_output = self.value_network.outputs

		self.Qvalue_network = tl.layers.InputLayer(action_output + value_output, name = 'Qvalue_input_origin')
		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = 300, act = tf.nn.relu, name = 'relu2_Qvalue_origin')
		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = 1, name = 'output_Qvalue_origin')

		self.Qvalue = self.Qvalue_network.outputs

	def build_Qvalue_network_target(self):

		self.value_input_target = tf.placeholder("float", [None, self.state_dim], name = 'value_input_target_ph')
		self.value_network_target = tl.layers.InputLayer(self.value_input_target, name = 'value_input_target')
		self.value_network_target = tl.layers.DenseLayer(self.value_network_target, n_units = 400, act = tf.nn.relu, name = 'value_relu1_target')
		self.value_network_target = tl.layers.DenseLayer(self.value_network_target, n_units = 300, name = 'value_output_target')


		self.action_input_target = tf.placeholder("float", [None, self.action_dim], name = 'action_input_target_ph')
		self.action_network_target = tl.layers.InputLayer(self.action_input_target, name = 'action_input_target')
		self.action_network_target = tl.layers.DenseLayer(self.action_network_target, n_units = 300, name = 'action_output_target')

		action_output_target = self.action_network_target.outputs
		value_output_target = self.value_network_target.outputs

		self.Qvalue_network_target = tl.layers.InputLayer(action_output_target + value_output_target, name = 'Qvalue_input_target')
		self.Qvalue_network_target = tl.layers.DenseLayer(self.Qvalue_network_target, n_units = 300, act = tf.nn.relu, name = 'relu2_Qvalue_target')
		self.Qvalue_network_target = tl.layers.DenseLayer(self.Qvalue_network_target, n_units = 1, name = 'output_Qvalue_target')

		self.Qvalue_target = self.Qvalue_network_target.outputs

	def build_policy_network(self):

		self.policy_input = tf.placeholder("float", [None, self.state_dim], name = 'policy_input_ph')

		self.policy_network = tl.layers.InputLayer(self.policy_input, name = 'policy_input')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 400, act = tf.nn.relu, name = 'relu1_policy')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 300, act = tf.nn.relu, name = 'relu2_policy')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = self.action_dim, act = tf.nn.tanh, name = 'output_policy')

		self.policy = self.policy_network.outputs

	def build_policy_network_target(self):

		self.policy_input_target = tf.placeholder("float", [None, self.state_dim], name = 'policy_input_target_ph')

		self.policy_network_target = tl.layers.InputLayer(self.policy_input_target, name = 'policy_input_target')
		self.policy_network_target = tl.layers.DenseLayer(self.policy_network_target, n_units = 400, act = tf.nn.relu, name = 'relu1_policy_target')
		self.policy_network_target = tl.layers.DenseLayer(self.policy_network_target, n_units = 300, act = tf.nn.relu, name = 'relu2_policy_target')
		self.policy_network_target = tl.layers.DenseLayer(self.policy_network_target, n_units = self.action_dim, act = tf.nn.tanh, name = 'output_policy_target')

		self.policy_target = self.policy_network_target.outputs

	def creat_training_method(self):

		self.q_gradients_input = tf.placeholder("float", [None, self.action_dim])
		self.target_input = tf.placeholder("float", [None, 1])

		self.score_input = tf.placeholder("float")
		self.replay_size = tf.placeholder("float")
		self.sigma_input = tf.placeholder("float")

		self.q_gradients = tf.gradients(self.Qvalue, self.action_input)

		value_cost = tf.reduce_mean(self.huber_loss(self.target_input, self.Qvalue, 100000.0))
		gradients = tf.gradients(self.policy, self.policy_network.all_params, -self.q_gradients_input)

		self.Qvalue_opt = tf.train.AdamOptimizer(VALUE_LR).minimize(value_cost)
		self.policy_opt = tf.train.AdamOptimizer(POLICY_LR).apply_gradients(zip(gradients, self.policy_network.all_params))

		value_summary = tf.reduce_mean(self.Qvalue, reduction_indices = 0)

		with tf.name_scope('cost'):
			tf.summary.scalar('value_cost', value_cost)

		with tf.name_scope('value'):
			tf.summary.scalar('score', self.score_input)

			for i in xrange(self.action_dim):
				tf.summary.scalar('value', value_summary[i])

		with tf.name_scope('param'):
			tf.summary.scalar('replay_size', self.replay_size)
			tf.summary.scalar('sigma', self.sigma_input)

	def perceive(self, state, action, reward, next_state, done):

		self.replay_buffer.append((state, action[0], reward, next_state, done))

		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_network()

		if done:
			self.reset_noise()

	def train_network(self):

		self.time_step += 1

		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		target_batch = []

		policy_target_batch = self.policy_target.eval(feed_dict = {self.policy_input_target:next_state_batch})

		Qvalue_target_batch = self.Qvalue_target.eval(feed_dict = {self.value_input_target:next_state_batch, self.action_input_target:policy_target_batch})

		for i in xrange(BATCH_SIZE):

			done = minibatch[i][4]

			if done:

				target_batch.append(reward_batch[i])

			else:

				target_batch.append(reward_batch[i] + GAMMA_NUM * Qvalue_target_batch[i])

		target_batch = np.asarray(target_batch)
		action_batch = np.asarray(action_batch)
		state_batch = np.asarray(state_batch)
		target_batch = np.resize(target_batch, [BATCH_SIZE, 1])

		feed_dict = {
		self.target_input:target_batch,
		self.action_input:action_batch,
		self.value_input:state_batch,
		self.score_input:self.score,
		self.replay_size:len(self.replay_buffer),
		self.sigma_input:self.sigma
		}

		summary, _ = self.session.run([self.merge_all, self.Qvalue_opt], feed_dict = feed_dict)

		self.writer.add_summary(summary, self.time_step)

		policy_batch = self.policy.eval(feed_dict = {self.policy_input:state_batch})

		q_gradients_batch = self.generate_q_gradients(state_batch, policy_batch)

		self.session.run(self.policy_opt, feed_dict = {self.policy_input:state_batch, self.q_gradients_input:q_gradients_batch})

		if self.time_step % 100 == 0:
			self.update_target_old()



		# self.update_target(self.Qvalue_network, self.Qvalue_network_target)
		# self.update_target(self.policy_network, self.policy_network_target)
		# self.update_target(self.value_network, self.value_network_target)
		# self.update_target(self.action_network, self.action_network_target)

	def update_target(self, origin_network, target_network):

		origin_params = origin_network.all_params
		target_params = target_network.all_params

		for i in xrange(len(origin_params)):
			tf.assign(target_params[i], (1 - TAU_NUM) * origin_params[i] + TAU_NUM * target_params[i])

		tl.files.assign_params(self.session, target_params, target_network)

	def update_target_old(self):

		tl.files.assign_params(self.session, self.value_network.all_params, self.value_network_target)
		tl.files.assign_params(self.session, self.action_network.all_params, self.action_network_target)
		tl.files.assign_params(self.session, self.Qvalue_network.all_params, self.Qvalue_network_target)
		tl.files.assign_params(self.session, self.policy_network.all_params, self.policy_network_target)

	def init_param(self, env_name):

		self.env_name = env_name

		self.replay_buffer = deque()

		env = gym.make(env_name)

		self.state_dim = len(env.observation_space.high)
		self.action_dim = len(env.action_space.high)

		self.action_high = env.action_space.high
		self.action_low = env.action_space.low

		print 'action_dim: ', self.action_dim, ' --- state_dim: ', self.state_dim

		env.close()

		self.time_step = 0
		self.score = 0
		self.sigma = INIT_SIGMA

		self.reset_noise()

	def huber_loss(self, y_true, y_pred, max_grad = 1.0):

		err = tf.abs(y_true - y_pred, name = 'abs')
		mg = tf.constant(max_grad, name = 'max_grad')

		lin = mg * (err - 0.5 * mg)
		quad = 0.5 * err * err

		return tf.where(err < mg, quad, lin)

	def generate_q_gradients(self, state_batch, action_batch):
		return self.session.run(self.q_gradients,feed_dict={
			self.value_input:state_batch,
			self.action_input:action_batch
			})[0]

	def get_action(self, state):

		state = state.reshape(1, self.state_dim)

		return self.policy.eval(feed_dict = {self.policy_input:state})


	def get_noise_action(self, state):

		state = state.reshape(1, self.state_dim)

		action = self.policy.eval(feed_dict = {self.policy_input:state})

		action = action + self.generate_noise()

		action = np.clip(action, self.action_low, self.action_high)

		return action

	def reset_noise(self):
		self.action_noise = np.ones(self.action_dim) * MU_NUM

	def generate_noise(self):
		x = self.action_noise
	
		if self.sigma < 0.01:
			self.sigma = 0.01

		dx = THETA_NUM * (MU_NUM - x) + self.sigma * np.random.randn(len(x))
		self.action_noise = x + dx

		return self.action_noise

	def write_score(self, score):
		self.score = score

	def real_action(self, action):
		return (action * self.action_high)


def train():

	env = gym.make(ENV_NAME)

	# env = wrappers.Monitor(env, '/tmp/E')

	agent = DDPG(ENV_NAME)

	for episode in xrange(100000):

		state = env.reset()
		score = 0

		for step in xrange(10000):

			action = agent.get_noise_action(state)

			next_state, reward, done, _ = env.step(agent.real_action(action))

			score += reward

			agent.perceive(state, action, reward, next_state, done)

			state = next_state

			if done:
				agent.write_score(score)

				break

		if episode % 100 == 0 and episode > 0:
			score = 0
			TRAIN_FLAG = False
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(1000000):
					action = agent.get_action(state)
					state, reward, done, _ = env.step(agent.real_action(action))
					score += reward
					if done:
						# if score < 9100:
						# 	TRAIN_FLAG = True
						break
				# if TRAIN_FLAG:
				# 	break

			# if TRAIN_FLAG:
			# 	continue

			score /= TEST
			print 'episode: ', episode, ' | score : ', score
			if score > 9100:
				break


if __name__ == '__main__':

	train()

 