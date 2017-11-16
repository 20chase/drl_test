import time
import random

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils as U


class PrioritizedDoubleDuelingDQN(object):
    def __init__(self, session, args, obs_dim, act_dim):
        self.sess = session
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if self.args.prioritized:
            self.buffer = U.PrioritizedReplayBuffer(
                self.args.buffer_size, alpha=self.args.alpha)
            self.speed_beta = (1. - self.args.beta) / self.args.max_steps
            self.beta = self.args.beta
        else:
            self.buffer = U.ReplayBuffer(self.args.buffer_size)

        self.time_step = 0
        self.score = 0
        self.epsilon = 1
        self.speed_eps = (1 - self.args.final_epsilon) / (self.args.explore_num)
        self._build_ph()
        self.eval_model, self.value_eval, self.adv_eval = self._build_network('eval')
        self.target_model, self.value_target, self.adv_target = self._build_network('target')
        self.mean_q, self.td_error, self.loss, self.opt, self.update_target = self._build_training_method()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tensorboard/dqn/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')
        self.weights_ph = tf.placeholder(tf.float32, [None, ], 'weights_ph')

    def _build_network(self, model_name):
        hid1_size = self.obs_dim * 10  
        hid3_size = self.act_dim * 5
        hid2_size = int(np.sqrt(hid1_size*hid3_size))

        network = tl.layers.InputLayer(self.obs_ph, name='input_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=hid1_size, act=tf.nn.relu, 
            name='{}_mlp1'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=hid2_size, act=tf.nn.relu, 
            name='{}_mlp2'.format(model_name))

        value_network = tl.layers.DenseLayer(network, n_units=hid3_size, act=tf.nn.relu, 
            name='{}_value'.format(model_name))
        value_network = tl.layers.DenseLayer(value_network, n_units=1, 
            name='{}_value_output'.format(model_name))

        adv_network = tl.layers.DenseLayer(network, n_units=self.act_dim, 
            name='{}_adv_output'.format(model_name))

        value = value_network.outputs
        adv = adv_network.outputs

        return [network, value_network, adv_network], value, adv

    def _build_training_method(self):
        mean_q = tf.reduce_mean(self.adv_eval) + tf.reduce_mean(self.value_eval)
        with tf.variable_scope('advantage'):
            adv = tf.reduce_sum(tf.multiply(self.adv_eval, self.act_ph), axis=1)
        with tf.variable_scope('q_value'):
            q_value = tf.squeeze(self.value_eval) + adv
        with tf.variable_scope('td_error'):
            td_error = q_value - self.ret_ph
        if self.args.huber:
            with tf.variable_scope('huber_loss'):
                errors = U.huber_loss(td_error)
        else:
            errors = tf.square(td_error)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(self.weights_ph*errors)

        # opt operation
        opt = tf.train.AdamOptimizer(self.args.lr).minimize(loss)

        # update target operation
        eval_vars = []
        target_vars = []

        for eval_net, target_net in zip(self.eval_model, self.target_model):
            eval_vars += eval_net.all_params
            target_vars += target_net.all_params

        update_target = []
        for var, var_target in zip(eval_vars, target_vars):
            update_target.append(var_target.assign(var))

        update_target = tf.group(*update_target)

        return mean_q, td_error, loss, opt, update_target

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        self.size_tb = tf.placeholder(tf.float32, name='size_tb')
        self.epsilon_tb = tf.placeholder(tf.float32, name='epsilon_tb')
        self.beta_tb = tf.placeholder(tf.float32, name='beta_tb')

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('params'):
            tf.summary.scalar('q-value', self.mean_q)
            tf.summary.scalar('score', self.score_tb)
            tf.summary.scalar('buffer_size', self.size_tb)
            tf.summary.scalar('epsilon', self.epsilon_tb)
            if self.args.prioritized:
                tf.summary.scalar('beta', self.beta_tb)

    def train(self):
        self.time_step += 1
        if self.args.prioritized:
            # sample experience from buffer
            self.beta += self.speed_beta
            experience = self.buffer.sample(self.args.batch_size, self.beta)
            (obses, acts, rews, new_obses, dones, weights, idxes) = experience
        else:
            obses, acts, rews, new_obses, dones = self.buffer.sample(self.args.batch_size)
            weights, idxes = np.ones_like(rews), None

        # compute rets
        adv_eval, adv_target, value_target = self.sess.run(
            [self.adv_eval, self.adv_target, self.value_target],
            feed_dict={self.obs_ph: new_obses})

        baselines = np.mean(adv_target, axis=1)

        rets = []
        for i in range(len(rews)):
            if dones[i]:
                rets.append(rews[i])
            else:
                rets.append(rews[i]+self.args.gamma*
                    (value_target[i][0]+adv_target[i][np.argmax(adv_eval[i])]-baselines[i]))

        rets = np.asarray(rets)

        # opt q-network
        feed_dict = {
        self.obs_ph: obses,
        self.act_ph: acts,
        self.ret_ph: rets,
        self.weights_ph: weights,
        self.score_tb: self.score,
        self.size_tb: len(self.buffer),
        self.epsilon_tb: self.epsilon
        }

        if self.args.prioritized:
            feed_dict[self.beta_tb] = self.beta

        summary, td_error,  _ = self.sess.run([self.merge_all, self.td_error, self.opt], 
            feed_dict=feed_dict)

        # write tensorboard file
        self.writer.add_summary(summary, self.time_step)
        
        # update target network
        if self.time_step % self.args.update_target_num == 0:
            self.sess.run(self.update_target)

        if self.args.prioritized:
            # update priorities
            new_priorities = np.abs(td_error) + 1e-6
            self.buffer.update_priorities(idxes, new_priorities)

    def action(self, obs, test=False):
        obs = np.reshape(obs, (1, self.obs_dim))
        feed_dict =  {self.obs_ph: obs}
        if self.args.test_alg or test:
            return np.argmax(self.sess.run(self.adv_eval, feed_dict)[0])
        elif self.time_step == 0:
            return random.randint(0, self.act_dim-1)

        # epsilon-greedy exploration
        self.epsilon -= self.speed_eps
        if self.epsilon < self.args.final_epsilon:
            self.epsilon = self.args.final_epsilon

        if random.random() <= self.epsilon:
            return random.randint(0, self.act_dim-1)
        else:
            return np.argmax(self.sess.run(self.adv_eval, feed_dict)[0])

    def get_score(self, score):
        # get cumulative rewards
        self.score = score

    def one_hot_key(self, act):
        one_hot_key = np.zeros(self.act_dim)
        one_hot_key[act] = 1.
        return one_hot_key

    def save_network(self, model_name):
        for i, network in enumerate(self.model):
            tl.files.save_npz(network.all_params, 
                name='../model/dqn/{}_{}.npz'.format(model_name, i),
                sess=self.sess)

    def load_network(self, model_name):
        for i, network in enumerate(self.model):
            params = tl.files.load_npz(
                name='../model/dqn/{}_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, network)