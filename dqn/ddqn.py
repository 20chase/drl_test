import gym

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from replay_buffer import PrioritizedReplayBuffer


class PrioritizedDoubleDQN(object):
    def __init__(self, session, parser, obs_dim, act_dim):
        self.sess = session
        self.parser = parser
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer = PrioritizedReplayBuffer(
            self.parser.buffer_size, alpha=self.parser.alpha)

        self._build_ph()
        self.eval_network, self.eval_q = self._build_network('eval')
        self.target_network, self.target_q = self._build_network('target')

        self._build_training_method()

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')

    def _build_network(self, model_name):
        hid1_size = self.obs_dim * 10  
        hid3_size = self.act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        network = tl.layers.InputLayer(self.obs_ph, name='input_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=hid1_size, act=tf.nn.relu, 
            name='relu1_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=hid2_size, act=tf.nn.relu, 
            name='relu2_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=hid3_size, act=tf.nn.relu, 
            name='relu3_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=self.act_dim, name='output_{}'.format)

        q_action = network.outputs

        return network, q_action

    def _build_training_method(self):
        q_value = tf.reduce_sum(tf.multiply(self.eval_q, self.act_ph), axis=1)
        td_error = tf.square(q_value-self.ret_ph)
        loss = tf.reduce_mean(td_error)

        self.opt = tf.train.AdamOptimizer(self.parser.lr).minimize(loss)

    def 


