import gym
import random

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from gym import wrappers
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.5 # final value of epsilon
EXPLOER_NUM = 10000
REPLAY_SIZE = 20000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
LEARNING_RATE = 5e-4
DECLAY_RATE = 1e-4
DECLAY_FLAG = True
DISPLAY = False
SAVE = False
LOAD = False
# MODE_NAME = 'LunarLander-v2'
MODE_NAME = 'CartPole-v1'
EPISODE = 10000 # Episode limitation
STEP = 10000 # Step limitation in an episode
TEST = 50

UPDATE_TIME = 500
OBSERVE_NUM = 64
TARGET_NUM = 995
EVAL_FLAG = False

class DDQN():
    def __init__(self, session, args, obs_dim, act_dim):
        self.session = session
        self.args = args
        self.buffer = deque()
        self.time_step = 0
        self.score = 0
        self.epsilon = 1.
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.speed_eps = (1 - self.args.final_epsilon) / (self.args.explore_num)

        self._build_ph()
        self.q_eval, self.network = self._build_network('eval')
        self.q_target, self.network_target = self._build_network('target')
        self._build_training_method()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/A', self.session.graph)
        self.session.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], name='obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], name='act_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None], name='ret_ph')

    def _build_network(self, model_name):
        network = tl.layers.InputLayer(self.obs_ph, name='input_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=64, act=tf.nn.relu, name='relu1_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=32, act=tf.nn.relu, name='relu2_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=16, act=tf.nn.relu, name='relu3_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=self.act_dim, name='output_{}'.format(model_name))

        q_value = network.outputs
        return q_value, network

    def _build_training_method(self):
        q_value = tf.reduce_sum(tf.multiply(self.q_eval, self.act_ph), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.ret_ph - q_value))
        self.opt = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)

    def add(self, obs, act, rew, new_obs, done):
        self.buffer.append((obs, act, rew, new_obs, done))
        if len(self.buffer) > REPLAY_SIZE:
            self.buffer.popleft()

        if len(self.buffer) > BATCH_SIZE:
            self.train()

    def get_score(self, score):
        self.score = score 

    def one_hot_key(self, act):
        one_hot_key = np.zeros(self.act_dim)
        one_hot_key[act] = 1.
        return one_hot_key

    def train(self):
        self.time_step += 1
        
        minibatch = random.sample(self.buffer, BATCH_SIZE)
        obses = [data[0] for data in minibatch]
        acts = [data[1] for data in minibatch]
        rews = [data[2] for data in minibatch]
        new_obses = [data[3] for data in minibatch]

        rets = []
        q_eval, q_target = self.session.run([self.q_eval, self.q_target], feed_dict={self.obs_ph: new_obses})
        
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                rets.append(rews[i])
            else :
                rets.append(rews[i] + self.args.gamma * q_target[i][np.argmax(q_eval[i])])

        replay_size = len(self.buffer)

        feed_dict = {
        self.ret_ph:rets,
        self.act_ph:acts,
        self.obs_ph:obses
        }

        self.session.run(self.opt, feed_dict=feed_dict)

    def update_target(self):
        tl.files.assign_params(self.session, self.network.all_params, self.network_target)

    def action(self, obs, test=False):
        obs = np.reshape(obs, (1, self.obs_dim))
        feed_dict =  {self.obs_ph: obs}
        if self.args.test_alg or test:
            return np.argmax(self.session.run(self.q_eval, feed_dict)[0])
        elif self.time_step == 0:
            return random.randint(0, self.act_dim-1)

        # epsilon-greedy exploration
        self.epsilon -= self.speed_eps
        if self.epsilon < self.args.final_epsilon:
            self.epsilon = self.args.final_epsilon

        if random.random() <= self.epsilon:
            return random.randint(0, self.act_dim-1)
        else:
            return np.argmax(self.session.run(self.q_eval, feed_dict)[0])
