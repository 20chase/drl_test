import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils as U

from collections import OrderedDict
from sklearn.utils import shuffle


class DiscreteA2C(object):
    def __init__(self, sess, args, obs_dim, act_dim):
        self.sess = sess
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.time_step = 0
        self.score = 0
        self.lr_scheduler = U.scheduler(v=self.args.lr,
            nvalues=self.args.max_steps, schedule='linear')

        self._build_ph()
        self.model, self.value, self.act_value, self.act = self._build_net()
        self.costs, self.opt = self._build_training_method()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tensorboard/a3c/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ], 'adv_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')
        self.lr_ph = tf.placeholder(tf.float32, name='lr_ph')

    def _build_net(self):
        hid1_size = self.obs_dim * 15
        hid2_size = self.obs_dim * 10

        main_net = tl.layers.InputLayer(self.obs_ph, name='input')
        main_net = tl.layers.DenseLayer(main_net, n_units=hid1_size, act=tf.nn.relu,
            name='main_relu1')
        main_net = tl.layers.DenseLayer(main_net, n_units=hid2_size, act=tf.nn.relu,
            name='main_relu2')
        act_net = tl.layers.DenseLayer(main_net, n_units=self.act_dim,
            name='output_act')
        value_net = tl.layers.DenseLayer(main_net, n_units=1, 
            name='output_value')

        value = value_net.outputs[:, 0]
        act_value = act_net.outputs
        with tf.variable_scope('sample_act'):
            act = U.sample(act_value)

        return [main_net, act_net, value_net], value, act_value, act

    def _build_training_method(self):
        with tf.variable_scope('td_error'):
            td_error = tf.squeeze(self.value) - self.ret_ph
        if self.args.huber:
            with tf.variable_scope('huber_loss'):
                errors = U.huber_loss(td_error)
        else:
            with tf.variable_scope('mse'):
                errors = tf.square(td_error)
        with tf.variable_scope('critic_loss'):
            critic_loss = tf.reduce_mean(errors)

        with tf.variable_scope('act_prob'):
            act_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.act_value, labels=self.act_ph)
        with tf.variable_scope('actor_loss'):
            actor_loss = tf.reduce_mean(self.adv_ph * act_prob)

        with tf.variable_scope('entropy'):
            entropy = tf.reduce_mean(U.cat_entropy(self.act_value))

        with tf.variable_scope('total_loss'):
            loss = actor_loss + 0.5 * critic_loss - 0.0 * entropy

        params = U.get_all_params(self.model)
        grads = tf.gradients(loss, params)
        if self.args.max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.args.max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr_ph, decay=0.99, epsilon=1e-5)
        opt = trainer.apply_gradients(grads)

        return [actor_loss, critic_loss, entropy, total_loss], opt

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        mean_v = tf.reduce_mean(self.value)

        with tf.name_scope('loss'):
            tf.summary.scalar('actor_loss', self.costs[0])
            tf.summary.scalar('critic_loss', self.costs[1])
            tf.summary.scalar('entropy', self.costs[2])
            tf.summary.scalar('total_loss', self.costs[3])

        with tf.name_scope('params'):
            tf.summary.scalar('score', self.score_tb)
            tf.summary.scalar('value', mean_v)

        for i in range(len(self.model)):
            for param in self.model[i].all_params:
                tf.summary.histogram(param.name, param)

    def update(self, obses, acts, rews, values):
        self.time_step += 1
        # compute advantages
        advs = rews - values
        # compute learning rate
        for step in range(len(rews)):
            lr = self.lr_scheduler.value()

        feed_dict = {
        self.obs_ph: obses,
        self.act_ph: acts,
        self.adv_ph: advs,
        self.ret_ph: rews,
        self.lr_ph: lr,
        self.score_tb: self.score
        }

        summary, actor_loss, critic_loss, entropy, total_loss, _ = self.sess.run(
            [self.merge_all, self.costs[0], self.costs[1], self.costs[2], self.costs[3], self.opt],
            feed_dict=feed_dict)

        self.writer.add_summary(summary self.time_step)

    def step(self, obs):
        feed_dict = {self.obs_ph: obs}
        act, value = self.sess.run([self.act, self.value], feed_dict=feed_dict)
        return act, value

    def get_value(self, obs):
        return self.sess.run(self.value, feed_dict={self.obs_ph: obs})

    def save_network(self, model_name):
        for i, network in enumerate(self.model):
            tl.files.save_npz(network.all_params, 
                name='../model/a3c/{}_{}.npz'.format(model_name, i),
                sess=self.sess)

    def load_network(self, model_name):
        for i, network in enumerate(self.model):
            params = tl.files.load_npz(
                name='../model/a3c/{}_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, network)

