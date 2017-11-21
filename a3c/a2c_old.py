import tensorflow as tf
import tensorlayer as tl
import gym

import numpy as np
import random
import os
from gym import wrappers
from collections import deque

# hyper parameter

VALUE_LR = 1e-3
POLICY_LR = 1e-4
GAMMA = 0.99 
MODE_NAME = 'CartPole-v1'
EPISODE = 10000
STEP = 10000
T_MAX = 30

class A2C():
	def __init__(self, env):

		self.time_step = 0
		self.score = 0
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n

		self.create_value_network()
		self.create_policy_network()
		self.create_training_method()

		# Init session
		self.session = tf.InteractiveSession()

		self.merged = tf.summary.merge_all()

		self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

		self.session.run(tf.global_variables_initializer())

	def create_value_network(self):

		self.value_input = tf.placeholder("float", [None, self.state_dim])
		self.value_network = tl.layers.InputLayer(self.value_input, name = 'Qvalue_input')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 400, act = tf.nn.relu, name = 'value_relu1')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 300, act = tf.nn.relu, name = 'value_relu2')
		# self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 64, act = tf.nn.relu, name = 'value_relu3')
		# self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 32, act = tf.nn.relu, name = 'value_relu4')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 1, name = 'value_output')

		self.value = self.value_network.outputs

	def create_policy_network(self):

		self.policy_input = tf.placeholder("float", [None, self.state_dim])
		self.policy_network = tl.layers.InputLayer(self.policy_input, name = 'policy_input')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 400, act = tf.nn.relu, name = 'policy_relu1')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 300, act = tf.nn.relu, name = 'policy_relu2')
		# self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 64, act = tf.nn.relu, name = 'policy_relu3')
		# self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 32, act = tf.nn.relu, name = 'policy_relu4')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = self.action_dim, name = 'policy_output')

		self.action = self.policy_network.outputs
		self.action_prob = tf.nn.softmax(self.action)

	def create_training_method(self):

		self.action_input = tf.placeholder("float", [None, self.action_dim])
		self.target_input = tf.placeholder("float", [None])

		self.epsilon_sum = tf.placeholder("float")
		self.score_input = tf.placeholder("float")

		policy_prob = tf.nn.softmax_cross_entropy_with_logits(logits = self.action, labels = self.action_input)
		advantage = self.target_input - self.value

		self.value_cost = tf.reduce_mean(tf.square(self.target_input - self.value))
		self.policy_cost = tf.reduce_mean(policy_prob * tf.stop_gradient(advantage))

		value_batch = tf.reduce_mean(self.value)

		with tf.name_scope('cost'):
			tf.summary.scalar('value_cost', self.value_cost)

		with tf.name_scope('value'):
			tf.summary.scalar('reward', self.score_input)
			tf.summary.scalar('value', value_batch)

		self.value_opt = tf.train.AdamOptimizer(VALUE_LR).minimize(self.value_cost)
		self.policy_opt = tf.train.AdamOptimizer(POLICY_LR).minimize(self.policy_cost)

	def train_network(self, state_batch, action_batch, reward_batch, final_reward):

		self.time_step += 1

		state_batch = np.asarray(state_batch)
		action_batch = np.asarray(action_batch)
		reward_batch = self.computer_reward(reward_batch, final_reward)
	
		feed_dict = {
		self.policy_input:state_batch,
		self.value_input:state_batch,
		self.target_input:reward_batch,
		self.action_input:action_batch,
		self.score_input:self.score
		}

		_, _, summary = self.session.run([self.policy_opt, self.value_opt, self.merged], feed_dict = feed_dict)

		self.train_writer.add_summary(summary, self.time_step)

	def computer_reward(self, origin_reward, final_reward):
		origin_reward = np.asarray(origin_reward)
		reward_batch = np.zeros_like(origin_reward, dtype = np.float32)		

		discounted_item = final_reward

		for i in reversed(xrange(len(origin_reward))):

			discounted_item = discounted_item * GAMMA + origin_reward[i]
			reward_batch[i] = discounted_item

		return reward_batch 

	def get_action(self, state):

		state = state.reshape(1, self.state_dim)
		action_prob = self.action_prob.eval(feed_dict={self.policy_input: state})
		return np.random.choice(range(self.action_dim), p=action_prob.flatten())

	def write_reward(self, reward_sum):
		self.score = reward_sum

	def one_hot_process(self, action):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		return one_hot_action


def train_game():
  
	env = gym.make(MODE_NAME)
	agent = A2C(env)

	for episode in xrange(EPISODE):
	
		state = env.reset()
		score = 0
		state_batch = []
		action_batch = []
		reward_batch = []
		final_reward = 0
	
		for step in xrange(STEP):
		
			action = agent.get_action(state) 
			next_state,reward,done,_ = env.step(action)
			score += reward

			state_batch.append(state)
			action_batch.append(agent.one_hot_process(action))
			reward_batch.append(reward)

			if step % T_MAX == 0 or done:
				if not done:
					final_reward = agent.value.eval(feed_dict = {agent.value_input: np.resize(next_state, [1, agent.state_dim])})[0]
				agent.train_network(state_batch, action_batch, reward_batch, final_reward)

				state_batch = []
				action_batch = []
				reward_batch = []
				final_reward = 0

			state = next_state

			if done:
				agent.write_reward(score)
				
				break

		if episode % 100 == 0 and episode > 0:
			
			score = 0

			for _ in xrange(100):

				state = env.reset()

				for step in xrange(STEP):

					action = agent.get_action(state)
					state, reward, done, _ = env.step(action)
					score += reward

					if done:
						break

			print 'episode: ', episode, '   | score: ', score / 100


    
if __name__ == '__main__':
	train_game()








