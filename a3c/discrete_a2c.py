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

        self._build_ph()
        self.model, self.value, self.act_value, self.act_prob = self._build_net()
        self.critic_loss, self.critic_opt, self.actor_loss, self.actor_opt = self._build_training_method()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tensorboard/a3c/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ] 'adv_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')

        self.obses_buffer = None
        self.acts_buffer = None
        self.advs_buffer = None
        self.rets_buffer = None

    def _build_net(self):
        hid1_size = self.obs_dim * 5
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        critic_net = tl.layers.InputLayer(self.obs_ph, name='critic_input')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=hid1_size, act=tf.nn.relu,
            name='critic_relu1')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=hid2_size, act=tf.nn.relu,
            name='critic_relu2')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=hid3_size, act=tf.nn.relu,
            name='critic_relu3')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=1, name='critic_output')

        hid1_size = self.obs_dim * 5
        hid3_size = self.act_dim * 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        actor_net = tl.layers.InputLayer(self.obs_ph, name='actor_input')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=hid1_size, act=tf.nn.relu,
            name='actor_relu1')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=hid2_size, act=tf.nn.relu,
            name='actor_relu2')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=hid3_size, act=tf.nn.relu,
            name='actor_relu3')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=self.act_dim, name='actor_output')

        value = critic_net.outputs
        act_value = actor_net.outputs
        act_prob = tf.nn.softmax(act_value)

        return [critic_net, actor_net], value, act_value, act_prob

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

        critic_opt = tf.train.AdamOptimizer(self.args.critic_lr).minimize(critic_loss)
        actor_opt = tf.train.AdamOptimizer(self.args.actor_lr).minimize(actor_loss)

        return critic_loss, critic_opt, actor_loss, actor_loss

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        mean_v = tf.reduce_mean(self.value)

        with tf.name_scope('loss'):
            tf.summary.scalar('critic_loss', self.critic_loss)
            tf.summary.scalar('actor_loss', self.actor_loss)

        with tf.name_scope('params'):
            tf,summary.scalar('score', self.score_tb)
            tf.summary.scalar('value', mean_v)

    def update(self, obses, acts, advs, rets):
        num_batches = max(obses.shape[0] // 256, 1)
        batch_size = obses.shape[0] // num_batches
        if self.obses_buffer is None:
            obses_train = obses
            acts_train = acts
            advs_train = advs
            rets_train = rets
        else:
            obses_train = np.concatenate([obses, self.obses_buffer])
            acts_train = np.concatenate([acts, self.acts_buffer])
            advs_train = np.concatenate([advs, self.advs_buffer])
            rets_train = np.concatenate([rets, self.rets_buffer])

        self.obses_buffer = obses
        self.acts_buffer = acts
        self.advs_buffer = advs
        self.rets_buffer = rets

        for e in range(self.args.train_epochs):
            obses_train, acts_train, advs_train, rets_train = shuffle(
                obses_train, acts_train, advs_train, rets_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs = obses_train[start:end, :]
                act = acts_train[start:end, :]
                adv = advs_train[start:end]
                ret = rets_train[start:end]
                feed_dict = {self.obs_ph: obs,
                             self.act_ph: act,
                             self.adv_ph: adv,
                             self.ret_ph: ret}
                self.sess.run([self.critic_opt, self.actor_opt], feed_dict=feed_dict)

    def show(self, obses, acts, advs, rets, rews):
        self.time_step += 1
        score = rews.sum() / self.args.batch_size

        feed_dict = {
        self.obs_ph: obses,
        self.act_ph: acts,
        self.adv_ph: advs,
        self.ret_ph: rets,
        self.score_tb: score
        }

        summary, actor_loss, critic_loss = self.sess.run([self.merge_all, self.actor_loss, self.critic_loss], 
            feed_dict=feed_dict)

        self.writer.add(summary, self.time_step)

        stats = OrderedDict()
        stats["Iteration"] = self.time_step
        stats["Episodes"] = self.time_step * self.args.batch_size
        stats["Score"] = score
        stats["MeanStep"] = rews / self.args.batch_size
        stats["ActorLoss"] = actor_loss
        stats["CriticLoss"] = critic_loss

        return stats

    def action(self, obs, test=False):
        obs = np.reshape(obs, (1, self.obs_dim))
        feed_dict =  {self.obs_ph: obs}
        act_prob = self.sess.run(self.act_prob, feed_dict=feed_dict)
        action = np.random.choice(range(self.act_dim), p=act_prob.flatten())
        return self.one_hot_key(action)

    def get_value(self, obses):
        return self.sess.run(self.value, feed_dict={self.obs_ph: obses})

    def one_hot_key(self, act):
        one_hot_key = np.zeros(self.act_dim)
        one_hot_key[act] = 1.
        return one_hot_key

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

