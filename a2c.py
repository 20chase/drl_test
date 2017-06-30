import tensorflow as tf
import tensorlayer as tl
import gym

import numpy as np
import random
import os
from gym import wrappers
from collections import deque

# hyper parameter

VALUE_LR = 1e-4
POLICY_LR = 1e-4
GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
REPLAY_SIZE = 40000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
DISPLAY = False
SAVE = False
LOAD = False
DECLAY_RATE = 1e-4
DECLAY_FLAG = True
# MODE_NAME = 'Breakout-ram-v0'
MODE_NAME = 'LunarLander-v2'
# MODE_NAME = 'CartPole-v1'
EPISODE = 1000000000000 # Episode limitation
STEP = 10000 # Step limitation in an episode
TEST = 100
OBSERVE_NUM = 100
TARGET_NUM = 200
EVAL_FLAG = False

class A2C():
	def __init__(self, env):

		# init experience replay
		self.replay_buffer = deque()
		# init some parameters
		self.epsilon = INITIAL_EPSILON
		self.time_step = 0
		self.reward = 0
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n

		self.create_Qvalue_network()
		self.create_policy_network()
		self.create_training_method()

		# Init session
		self.session = tf.InteractiveSession()

		self.merged = tf.summary.merge_all()

		self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

		self.session.run(tf.global_variables_initializer())

	def create_Qvalue_network(self):

		self.Qvalue_input = tf.placeholder("float", [None, self.state_dim])

		self.Qvalue_network = tl.layers.InputLayer(self.Qvalue_input, name = 'Qvalue_input')
		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = 128, act = tf.nn.relu, name = 'Qvalue_relu1')
		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = 128, act = tf.nn.relu, name = 'Qvalue_relu2')
		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = 64, act = tf.nn.relu, name = 'Qvalue_relu3')
		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = 32, act = tf.nn.relu, name = 'Qvalue_relu4')

		self.Qvalue_network = tl.layers.DenseLayer(self.Qvalue_network, n_units = self.action_dim, name = 'Qvalue_output')

		self.Qvalue = self.Qvalue_network.outputs

	def create_policy_network(self):

		self.policy_input = tf.placeholder("float", [None, self.state_dim])

		self.policy_network = tl.layers.InputLayer(self.policy_input, name = 'policy_input')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 128, act = tf.nn.relu, name = 'policy_relu1')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 128, act = tf.nn.relu, name = 'policy_relu2')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 64, act = tf.nn.relu, name = 'policy_relu3')
		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = 32, act = tf.nn.relu, name = 'policy_relu4')

		self.policy_network = tl.layers.DenseLayer(self.policy_network, n_units = self.action_dim, name = 'policy_output')


		self.Policy = self.policy_network.outputs
		self.policy_prob = tf.nn.softmax(self.Policy)

	def create_training_method(self):

		self.action_input = tf.placeholder("float", [None, self.action_dim])
		self.target_input = tf.placeholder("float", [None])
		self.current_Qvalue = tf.placeholder("float", [None, self.action_dim])

		self.epsilon_sum = tf.placeholder("float")
		self.reward_sum = tf.placeholder("float")
		self.replay_size = tf.placeholder("float")

		Qvalue = tf.reduce_sum(tf.multiply(self.Qvalue, self.action_input), reduction_indices = 1)
		policy_prob = tf.nn.softmax_cross_entropy_with_logits(logits = self.Policy, labels = self.action_input)
		advantage_batch = tf.reduce_sum(tf.multiply(self.current_Qvalue, self.action_input), reduction_indices = 1) - tf.reduce_mean(self.current_Qvalue, reduction_indices = 1)

		self.value_cost = tf.reduce_mean(tf.square(self.target_input - Qvalue))
		self.policy_cost = tf.reduce_sum(tf.multiply(policy_prob, tf.stop_gradient(advantage_batch)))

		Qvalue_batch = tf.reduce_mean(tf.reduce_mean(self.Qvalue, axis = 0))

		with tf.name_scope('cost'):
			tf.summary.scalar('value_cost', self.value_cost)

		with tf.name_scope('value'):
			tf.summary.scalar('reward', self.reward_sum)
			tf.summary.scalar('Qvalue', Qvalue_batch)
			# tf.summary.scalar('Qvalue', Qvalue_batch[1])
			# tf.summary.scalar('Qvalue', Qvalue_batch[2])

		with tf.name_scope('param'):
			tf.summary.scalar('epsilon', self.epsilon_sum)
			tf.summary.scalar('replay_size', self.replay_size)

		self.value_opt = tf.train.AdamOptimizer(VALUE_LR).minimize(self.value_cost)
		self.policy_opt = tf.train.AdamOptimizer(POLICY_LR).minimize(self.policy_cost)

	def perceive(self,state,action,reward,next_state,done):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state,one_hot_action,reward,next_state,done))

		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Qvalue_network()

	def train_Qvalue_network(self):
		self.time_step += 1
		
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		target_batch = []
		
		# current_Qvalue_batch = np.sum(np.multiply(current_Qvalue_batch, action_batch), axis = 1)

		next_Qvalue_batch = self.Qvalue.eval(feed_dict={self.Qvalue_input:next_state_batch})

		action_prob = self.policy_prob.eval(feed_dict={self.policy_input: next_state_batch})

		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				target_batch.append(reward_batch[i])
			else :
				# one_hot_action = np.argmax(action_prob[i])
				one_hot_action = np.random.choice(range(self.action_dim), p=action_prob[i].flatten())
				target_batch.append(reward_batch[i] + GAMMA * next_Qvalue_batch[i][one_hot_action])

		replay_size = len(self.replay_buffer)

		feed_dict = {
		self.target_input:target_batch,
		self.action_input:action_batch,
		self.Qvalue_input:state_batch,
		self.reward_sum:self.reward,
		self.epsilon_sum:self.epsilon,
		self.replay_size:replay_size
		}

		summary, _ = self.session.run([self.merged, self.value_opt], feed_dict = feed_dict)

		self.train_writer.add_summary(summary, self.time_step)

	def train_policy_network(self, state_batch, action_batch):

		state_batch = np.asarray(state_batch)
		action_batch = np.asarray(action_batch)
	
		current_Qvalue_batch = self.Qvalue.eval(feed_dict = {self.Qvalue_input:state_batch})

		feed_dict = {
		self.policy_input:state_batch,
		self.current_Qvalue:current_Qvalue_batch,
		self.action_input:action_batch
		}

		self.session.run(self.policy_opt, feed_dict = feed_dict)

	def get_action(self, state):

		if self.time_step < OBSERVE_NUM:
			return random.randint(0,self.action_dim - 1)

		if DECLAY_FLAG:
			self.epsilon *= (1 - DECLAY_RATE)
		else:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLOER_NUM

		if self.epsilon < FINAL_EPSILON:
			self.epsilon *= (1 - DECLAY_RATE * 0.01)

		if random.random() <= self.epsilon:
			return random.randint(0,self.action_dim - 1)
		else:
			state = state.reshape(1, self.state_dim)

			action_prob = self.policy_prob.eval(feed_dict={self.policy_input: state})

			return np.random.choice(range(self.action_dim), p=action_prob.flatten())

	def write_reward(self, reward_sum):
		self.reward = reward_sum

	def one_hot_process(self, action):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		return one_hot_action


def train_game():
  
	env = gym.make(MODE_NAME)
	if EVAL_FLAG:
		env = wrappers.Monitor(env, '/tmp/' + MODE_NAME)

	agent = A2C(env)

	if LOAD is True:
		value_params = tl.files.load_npz(name = MODE_NAME + '_value.npz')
		policy_params = tl.files.load_npz(name = MODE_NAME + '_policy.npz')
		tl.files.assign_params(agent.session, value_params, agent.Qvalue_network)
		tl.files.assign_params(agent.session, policy_params, agent.policy_network)

	reward_mean = 0
	reward_sum = 0
	end_flag = False
	for episode in xrange(EPISODE):
	# initialize task
		state = env.reset()
		if end_flag:
			break

		state_batch = []
		action_batch = []
		
		# Train
		for step in xrange(STEP):
			if DISPLAY is True:
				env.render()
			action = agent.get_action(state) 
			next_state,reward,done,_ = env.step(action)
			reward_sum += reward
			agent.perceive(state,action,reward,next_state,done)

			state_batch.append(state)
			action_batch.append(agent.one_hot_process(action))
		
			state = next_state

			if done:

				agent.write_reward(reward_sum)

				agent.train_policy_network(state_batch, action_batch)

				state_batch = []
				action_batch = []

				reward_mean += reward_sum
				
				print 'epsido: ', episode, '... reward_sum: ', reward_sum

				reward_sum = 0

				if episode % TEST == 0:
					if SAVE is True:
						tl.files.save_npz(agent.Qvalue_network.all_params, name=MODE_NAME + '_value.npz')
						tl.files.save_npz(agent.policy_network.all_params, name=MODE_NAME + '_policy.npz')
					reward_mean /= (TEST + 1)

					if (reward_mean > TARGET_NUM):
						end_flag = True


					print 'episode:', episode, '   reward_mean:', reward_mean
				break
    

if __name__ == '__main__':
	train_game()
	if EVAL_FLAG:
		gym.upload('/tmp/' + MODE_NAME, api_key='sk_nXYWtyR0CfjmTgSiJVJA')







