import time
import joblib

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils as U


class DPPO(object):
    def __init__(self, sess, args, ob_space, ac_space):
        self.sess = sess
        self.args = args
        self.obs_dim = ob_space.shape[0]
        self.act_dim = ac_space.shape[0]

        self.time_step = 0
        self.score = 0
        self.eta = 1.

        self._build_ph()
        self.model, self.means, self.value, self.pd, self.act, self.logp = self._build_net()
        self.costs, self.approx_kl, self.opt = self._build_training_method()

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ], 'adv_ph')

        self.old_vpred_ph = tf.placeholder(tf.float32, [None, ], 'old_vpred_ph')
        self.old_logp_ph = tf.placeholder(tf.float32, [None, ], 'old_logp_ph')

        self.beta_ph = tf.placeholder(tf.float32, name='beta_ph')
        self.lr_ph = tf.placeholder(tf.float32, name='lr_ph')

    def _build_net(self):
        actor_net = tl.layers.InputLayer(self.obs_ph, name='actor_input')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=64, act=tf.nn.tanh,
            name='actor_tanh1')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=64, act=tf.nn.tanh,
            name='actor_tanh2')
        actor_net = tl.layers.DenseLayer(actor_net, n_units=self.act_dim, act=tf.nn.tanh,
            name='act_output')

        critic_net = tl.layers.InputLayer(self.obs_ph, name='critic_input')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=64, act=tf.nn.tanh,
            name='critic_tanh1')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=64, act=tf.nn.tanh,
            name='critic_tanh2')
        critic_net = tl.layers.DenseLayer(critic_net, n_units=1, name='value_output')

        logstd = tf.get_variable(name='logstd', shape=[1, self.act_dim],
            initializer=tf.zeros_initializer())

        means = actor_net.outputs
        value = critic_net.outputs

        pdparam = tf.concat([means, means*0.0 + logstd], axis=1)

        pdtype = U.DiagGaussianPdType(self.act_dim)
        pd = pdtype.pdfromflat(pdparam)

        with tf.variable_scope('sample_act'):
            act = pd.sample()

        logp = pd.neglogp(act)

        return [actor_net, critic_net], means, value[:, 0], pd, act, logp

    def _build_training_method(self):
        logp = self.pd.neglogp(self.act_ph)
        with tf.variable_scope('approx_kl'):
            approx_kl = .5 * tf.reduce_mean(tf.square(logp-self.old_logp_ph))
        with tf.variable_scope('pg_loss'):
            ratio = tf.exp(self.old_logp_ph-logp)
            pg_loss1 = tf.reduce_mean(-self.adv_ph * ratio)
            pg_loss2 = tf.reduce_mean(self.beta_ph * approx_kl)
            pg_loss3 = self.eta * tf.square(tf.maximum(0.0, approx_kl - 2.*self.args.d_targ))
            pg_loss = pg_loss1 + pg_loss2 + pg_loss3

        vpred = self.value
        with tf.variable_scope('vf_loss'):
            vf_loss = .5 * tf.reduce_mean(tf.square(vpred - self.ret_ph))

        with tf.variable_scope('entropy'):
            entropy = tf.reduce_mean(self.pd.entropy())

        with tf.variable_scope('total_loss'):
            loss = pg_loss - self.args.ent_coef * entropy

        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if self.args.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.args.max_grad_norm)

        grads = list(zip(grads, params))
        opt_list = []
        trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        vf_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(vf_loss)
        pg_opt = trainer.apply_gradients(grads)
        opt_list.append(vf_opt)
        opt_list.append(pg_opt)
        opt = tf.group(*opt_list)

        return [pg_loss, vf_loss, entropy, loss], approx_kl, opt

    def train(self, lr, beta, obses, rets, acts, values, logps):
        # compute advantage
        advs = rets - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        feed_dict = {
        self.obs_ph: obses,
        self.act_ph: acts,
        self.adv_ph: advs,
        self.ret_ph: rets,
        self.old_vpred_ph: values,
        self.old_logp_ph: logps,
        self.lr_ph: lr,
        self.beta_ph: beta
        }

        return self.sess.run(
            [self.costs[0], self.costs[1], self.costs[2], 
            self.approx_kl, self.beta_ph, self.opt], feed_dict=feed_dict)[:-1]

    def step(self, obs):
        feed_dict = {self.obs_ph: obs}
        act, value, logp = self.sess.run([self.act, self.value, self.logp],
            feed_dict=feed_dict)
        return act, value, logp

    def get_value(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.value, feed_dict=feed_dict)

    def get_action(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.means, feed_dict=feed_dict)

    def save_network(self, save_path):
        params = tf.trainable_variables()
        ps = self.sess.run(params)
        joblib.dump(ps, save_path)

    def load_network(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = tf.trainable_variables()
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)