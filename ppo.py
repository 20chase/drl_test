#! /usr/bin/env python3
import gym
import scipy.signal

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from gym import wrappers
from sklearn.utils import shuffle


# hyper parameters

KL_TARGET = 0.003
GAMMA = 0.995
LAMBDA = 0.98
BATCH_SIZE = 10
MAX_EPISODES = 10000

ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'Humanoid-v1'
# ENV_NAME = 'InvertedPendulum-v1'

class PPO(object):
	def __init__(self, env):
		self.init_param(env)
		self.session = tf.InteractiveSession()

		self._build_actor()
		self._build_critic()
		self._build_summary()

		self.merge_all = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter('/tmp/ppo', self.session.graph)
		self.session.run(tf.global_variables_initializer())

	def _build_actor(self):
		# obs, act, adv
		self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
		self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
		self.advantages_ph = tf.placeholder(tf.float32, [None,], 'advantages_ph')
		# old_pi
		self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
		self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
		# param
		self.beta_ph = tf.placeholder(tf.float32, name = 'beta')
		self.eta_ph = tf.placeholder(tf.float32, name = 'eta')
		self.lr_ph = tf.placeholder(tf.float32, name = 'lr')
		# network
		hid1_size = self.obs_dim * 15  
		hid3_size = self.act_dim * 15
		hid2_size = int(np.sqrt(hid1_size * hid3_size))
		# self.actor_lr = float(9e-4) / float(np.sqrt(hid2_size))
		self.actor_lr = 3e-4

		# out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
		# 	kernel_initializer=tf.random_normal_initializer(
		# 	stddev=np.sqrt(1.0 / self.obs_dim)), name="h1")
		# out = tf.layers.dense(out, hid2_size, tf.tanh,
		# 	kernel_initializer=tf.random_normal_initializer(
		# 	stddev=np.sqrt(1.0 / float(hid1_size))), name="h2")
		# out = tf.layers.dense(out, hid3_size, tf.tanh,
		# 	kernel_initializer=tf.random_normal_initializer(
		# 	stddev=np.sqrt(1.0 / float(hid2_size))), name="h3")
		# self.means = tf.layers.dense(out, self.act_dim,
		# 							 kernel_initializer=tf.random_normal_initializer(
		# 								 stddev=np.sqrt(1.0 / float(hid3_size))), name="means")

		self.network = tl.layers.InputLayer(self.obs_ph, name = 'network_input')
		self.network = tl.layers.DenseLayer(self.network, n_units = hid1_size, act = tf.nn.tanh, 
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / self.obs_dim)), name = 'hide1')
		self.network = tl.layers.DenseLayer(self.network, n_units = hid2_size, act = tf.nn.tanh,
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid1_size))), name = 'hide2')
		self.network = tl.layers.DenseLayer(self.network, n_units = hid3_size, act = tf.nn.tanh, 
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid2_size))), name = 'hide3')
		self.mean_network = tl.layers.DenseLayer(self.network, n_units = self.act_dim, act = tf.nn.tanh,
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid3_size))), name = 'mean')

		self.means = self.mean_network.outputs

		logvar_speed = (10 * hid3_size) // 48
		log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
								   tf.constant_initializer(0.0))
		self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0

		# logprob
		self.logp = -0.5 * tf.reduce_sum(self.log_vars) + -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) / tf.exp(self.log_vars), axis=1)
		self.logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph) + -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) / tf.exp(self.old_log_vars_ph), axis=1)

		# kl and entropy
		with tf.variable_scope('kl'):
			self.kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars)) + 
				tf.reduce_sum(tf.square(self.means - self.old_means_ph) / tf.exp(self.log_vars), axis=1) -
				self.act_dim +
				tf.reduce_sum(self.log_vars) - tf.reduce_sum(self.old_log_vars_ph))

		with tf.variable_scope('entropy'):
			self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
								  tf.reduce_sum(self.log_vars))
		with tf.variable_scope('sample_action'):
			self.sampled_act = (self.means +
								tf.exp(self.log_vars / 2.0) *
								tf.random_normal(shape=(self.act_dim,)))

		with tf.variable_scope('actor_loss'):
			loss1 = -tf.reduce_mean(self.advantages_ph *
									tf.exp(self.logp - self.logp_old))
			loss2 = tf.reduce_mean(self.beta_ph * self.kl)
			loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
			self.actor_loss = loss1 + loss2 + loss3

		self.actor_opt = tf.train.AdamOptimizer(self.lr_ph).minimize(self.actor_loss)

	def _build_critic(self):
		self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')

		# network
		hid1_size = self.obs_dim * 15  
		hid3_size = 5  
		hid2_size = int(np.sqrt(hid1_size * hid3_size))
		# self.critic_lr = float(1e-2) / float(np.sqrt(hid2_size)) 
		self.critic_lr = 1e-3

		self.value_network = tl.layers.InputLayer(self.obs_ph, name = 'value_network_input')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = hid1_size, act = tf.nn.tanh, 
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / self.obs_dim)), name = 'value1')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = hid2_size, act = tf.nn.tanh,
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid1_size))), name = 'value2')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = hid3_size, act = tf.nn.tanh, 
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid2_size))), name = 'value3')
		self.value_network = tl.layers.DenseLayer(self.value_network, n_units = 1, act = tf.nn.tanh,
			W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid3_size))), name = 'value')

		self.value = self.value_network.outputs

		# loss
		with tf.variable_scope('critic_loss'):
			self.critic_loss = tf.reduce_mean(tf.square(tf.squeeze(self.value) - self.ret_ph))
		self.critic_opt = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

	def _build_summary(self):
		self.score_ph = tf.placeholder(tf.float32, name = 'score_ph')
		with tf.name_scope('loss'):
			tf.summary.scalar('actor_loss', self.actor_loss)
			tf.summary.scalar('critic_loss', self.critic_loss)

		with tf.name_scope('param'):
			tf.summary.scalar('entropy', self.entropy)
			tf.summary.scalar('kl', self.kl)
			tf.summary.scalar('beta', self.beta_ph)

		tf.summary.scalar('score', self.score_ph)


	def update_actor(self, obs, acts, advs):
		feed_dict = {
		self.obs_ph: obs,
		self.act_ph: acts,
		self.advantages_ph: advs,
		self.beta_ph: self.beta,
		self.eta_ph: self.eta,
		self.lr_ph: self.actor_lr * self.lr_multiplier
		}

		old_means_np, old_log_vars_np = self.session.run([self.means, self.log_vars],
													  feed_dict)
		feed_dict[self.old_log_vars_ph] = old_log_vars_np
		feed_dict[self.old_means_ph] = old_means_np

		for e in range(self.actor_epochs):
			self.session.run(self.actor_opt, feed_dict)
			kl = self.session.run(self.kl, feed_dict)
			if kl > self.kl_targ * 4: # early stopping
				break

		# magic setting
		if kl > self.kl_targ * 2:
			self.beta = np.minimum(35, 1.5 * self.beta)
			if self.beta > 30 and self.lr_multiplier > 0.1:
				self.lr_multiplier /= 1.5
		elif kl < self.kl_targ / 2.0:
			self.beta = np.maximum(1.0 / 35.0, self.beta / 1.5)
			if self.beta < (1.0 / 30.0) and self.lr_multiplier < 10:
				self.lr_multiplier *= 1.5

	def update_critic(self, x, y):
		num_batches = max(x.shape[0] // 256, 1)
		batch_size = x.shape[0] // num_batches
		if self.replay_buffer_x is None:
			x_train, y_train = x, y
		else:
			x_train = np.concatenate([x, self.replay_buffer_x])
			y_train = np.concatenate([y, self.replay_buffer_y])
		self.replay_buffer_x = x
		self.replay_buffer_y = y
		for e in range(self.critic_epochs):
			x_train, y_train = shuffle(x_train, y_train)
			for j in range(num_batches):
				start = j * batch_size
				end = (j + 1) * batch_size
				obs = x_train[start:end, :]
				ret = y_train[start:end]
				feed_dict = {self.obs_ph: obs,
							 self.ret_ph: ret}
				self.session.run(self.critic_opt, feed_dict=feed_dict)

	# def visualize(self, obs, acts, advs, old_means, old_stds, rets):
	# 	self.time_step += 1 
	# 	feed_dict = {
	# 	self.obs_ph: obs,
	# 	self.act_ph: acts,
	# 	self.advantages_ph: advs,
	# 	self.beta_ph: self.beta,
	# 	self.eta_ph: self.eta,
	# 	self.ret_ph: rets,
	# 	self.score_ph: self.score
	# 	}

	# 	summary, kl, entropy = self.session.run([self.merge_all, self.kl, self.entropy], feed_dict)
	# 	self.writer.add_summary(summary, self.time_step)

	def sample(self, obs):
		"""Draw sample from policy distribution"""
		obs = np.reshape(obs, (1, self.obs_dim))
		feed_dict = {self.obs_ph: obs}

		return self.session.run(self.sampled_act, feed_dict=feed_dict)

	def get_value(self, obs):
		values = self.value.eval(feed_dict = {self.obs_ph: obs})
		return values

	def convert_action(self, action):
		return action * self.act_high

	def init_param(self, env):
		# env param
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]
		self.act_high = env.action_space.high

		# value init
		self.time_step = 0
		self.score = 0

		# actor param
		self.beta = 1
		self.eta = 50
		self.kl_targ = KL_TARGET
		self.actor_epochs = 30
		self.actor_lr = None
		self.lr_multiplier = 1.0

		# critic param
		self.replay_buffer_x = None
		self.replay_buffer_y = None
		self.critic_epochs = 10
		self.critic_lr = None

	def write_score(self, score):
		self.score = score


def run_episode(env, agent, animate=False):
	state = env.reset()
	obs, acts, rewards = [], [], []
	done = False
	while not done:
		if animate:
			env.render()
		obs.append(state)
		action = agent.sample(state).reshape((1, -1)).astype(np.float32)
		acts.append(action)
		state, reward, done, _ = env.step(np.squeeze(action, axis=0))
		rewards.append(reward)

	return (np.asarray(obs), np.asarray(acts), np.asarray(rewards))

def run_policy(env, agent, episodes):
	trajectories = []
	for e in range(episodes):
		obs, acts, rewards = run_episode(env, agent)
		acts = np.reshape(acts, (len(rewards), agent.act_dim))
		trajectory = {
		'obs': obs,
		'acts': acts,
		'rewards': rewards
		}
		trajectories.append(trajectory)

	score = np.mean([t['rewards'].sum() for t in trajectories])
	return trajectories, score

def discount(x, gamma):
	return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def add_disc_sum_rew(trajectories, gamma):
	for trajectory in trajectories:
		rewards = trajectory['rewards'] * (1 - gamma)
		disc_sum_rew = discount(rewards, gamma)
		trajectory['disc_sum_rew'] = disc_sum_rew

def add_value(trajectories, agent):
	for trajectory in trajectories:
		obs = trajectory['obs']
		values = agent.get_value(obs)
		trajectory['values'] = np.squeeze(np.asarray(values))

def add_gae(trajectories, gamma, lam):
	for trajectory in trajectories:
		rewards = trajectory['rewards'] * (1 - gamma)
		values = trajectory['values']
		# temporal differences
		tds = rewards - values + np.append(values[1:] * gamma, 0)
		advantages = discount(tds, gamma * lam)
		advs = np.asarray(advantages)
		trajectory['advs'] = advs

def build_train_set(trajectories):
	observes = np.concatenate([t['obs'] for t in trajectories])
	actions = np.concatenate([t['acts'] for t in trajectories])
	disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
	advantages = np.concatenate([t['advs'] for t in trajectories])
	# normalize advantages
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

	return observes, actions, advantages, disc_sum_rew

def train():
	env = gym.make(ENV_NAME)
	agent = PPO(env)
	# env = wrappers.Monitor(env, '/tmp/A', force=True)
	run_policy(env, agent, episodes=5)
	e = 0
	while e < (MAX_EPISODES):
		trajectories, score = run_policy(env, agent, episodes = BATCH_SIZE)
		e += len(trajectories)
		add_value(trajectories, agent)
		add_disc_sum_rew(trajectories, GAMMA)
		add_gae(trajectories, GAMMA, LAMBDA)
		obs, acts, advs, rets = build_train_set(trajectories)
		# print '~~~~~~~ shape ~~~~~~~~~'
		# print 'obs: ', np.shape(obs)
		# print 'acts: ', np.shape(acts)
		# print 'advs: ', np.shape(advs)
		# print 'rets: ', np.shape(rets)

		agent.update_actor(obs, acts, advs)
		agent.update_critic(obs, rets)
		print (' ******** episode ', e, ' ********')
		print ('  score: ', score)
		print ('  beta:  ', agent.beta)
		print ('  lr:    ', agent.lr_multiplier * agent.actor_lr)



if __name__ == "__main__":
	train()












