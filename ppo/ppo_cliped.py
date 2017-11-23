import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils as U


class PPOCliped(object):
    def __init__(self, sess, args, ob_space, ac_space):
        self.sess = sess
        self.args = args
        self.obs_dim = ob_space.shape[0]
        self.act_dim = ac_space.shape[0]

        self.time_step = 0
        self.score = 0

        self._build_ph()
        self.model, self.means, self.value, self.pd, self.act, self.logp = self._build_net()
        self.costs, self.approx_kl, self.clip_frac, self.opt = self._build_training_method()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tensorboard/ppo/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ], 'adv_ph')

        self.old_vpred_ph = tf.placeholder(tf.float32, [None, ], 'old_vpred_ph')
        self.old_logp_ph = tf.placeholder(tf.float32, [None, ], 'old_logp_ph')

        self.lr_ph = tf.placeholder(tf.float32, name='lr_ph')
        self.clip_range_ph = tf.placeholder(tf.float32, name='clip_range_ph')

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

        return [actor_net, critic_net], means, value, pd, act, logp

    def _build_training_method(self):
        vpred = self.value
        vpred_clip = self.old_vpred_ph + tf.clip_by_value(self.value-self.old_vpred_ph,
            -self.clip_range_ph, self.clip_range_ph)
        with tf.variable_scope('vf_loss1'):
            vf_loss1 = tf.square(vpred-self.ret_ph)
        with tf.variable_scope('vf_loss2'):
            vf_loss2 = tf.square(vpred_clip-self.ret_ph)
        with tf.variable_scope('vf_loss'):
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))

        logp = self.pd.neglogp(self.act_ph)
        with tf.variable_scope('ratio'):
            ratio = tf.exp(self.old_logp_ph-logp)
        with tf.variable_scope('pg_loss1'):
            pg_loss1 = -self.adv_ph * ratio
        with tf.variable_scope('pg_loss2'):
            pg_loss2 = -self.adv_ph * tf.clip_by_value(ratio, 1.0-self.clip_range_ph, 1.0+self.clip_range_ph)
        with tf.variable_scope('pg_loss'):
            pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

        with tf.variable_scope('entropy'):
            entropy = tf.reduce_mean(self.pd.entropy())

        with tf.variable_scope('approx_kl'):
            approx_kl = .5 * tf.reduce_mean(tf.square(logp-self.old_logp_ph))
        with tf.variable_scope('clip_frac'):
            clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio-1.),self.clip_range_ph)))

        with tf.variable_scope('total_loss'):
            loss = pg_loss - self.args.ent_coef * entropy + self.args.vf_coef * vf_loss 

        # params = U.get_all_params(self.model)
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if self.args.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.args.max_grad_norm)

        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        opt = trainer.apply_gradients(grads)

        return [pg_loss, vf_loss, entropy, loss], approx_kl, clip_frac, opt

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        mean_v = tf.reduce_mean(self.value)

        with tf.name_scope('loss'):
            tf.summary.scalar('pg_loss', self.costs[0])
            tf.summary.scalar('vf_loss', self.costs[1])
            tf.summary.scalar('entropy', self.costs[2])
            tf.summary.scalar('total_loss', self.costs[3])

        with tf.name_scope('params'):
            tf.summary.scalar('score', self.score_tb)
            tf.summary.scalar('value', mean_v)
            tf.summary.scalar('approx_kl', self.approx_kl)
            tf.summary.scalar('clip_frac', self.clip_frac)
            tf.summary.scalar('lr', self.lr_ph)
            tf.summary.scalar('clip_range', self.clip_range_ph)

        for i in range(len(self.model)):
            for param in self.model[i].all_params:
                tf.summary.histogram(param.name, param)

    def learn(self, traj):
        self.time_step += 1
        obses = traj[0]
        acts = traj[1]
        rets = traj[2]
        values = traj[3]
        logps = traj[4]

        lr = self.args.lr
        clip_range = self.args.clip_range

        training_size = self.args.num_steps // self.args.num_batchs

        idxs = np.arange(self.args.num_steps)
        for _ in range(self.args.num_opts):
            np.random.shuffle(idxs)
            for start in range(0, self.args.num_steps, training_size):
                end = start + training_size
                idx = idxs[start:end]
                slices = (arr[idx] for arr in (obses, acts, rets, values, logps))
                self.update(lr, clip_range, *slices)

        self.show(lr, clip_range, obses, acts, rets, values, logps)

    def update(self, lr, clip_range, obses, acts, rets, values, logps):
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
        self.clip_range_ph: clip_range
        }

        self.sess.run(self.opt, feed_dict=feed_dict)

    def show(self, lr, clip_range, obses, acts, rets, values, logps):
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
        self.clip_range_ph: clip_range,
        self.score_tb: self.score
        }

        summary, pg_loss, vf_loss, entropy, total_loss, approx_kl, clip_frac = self.sess.run(
            [self.merge_all, self.costs[0], self.costs[1], self.costs[2], self.costs[3],
             self.approx_kl, self.clip_frac], feed_dict=feed_dict)

        self.writer.add_summary(summary, self.time_step)

        # return [pg_loss, vf_loss, entropy, total_loss, approx_kl, clip_frac]

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

    def save_network(self, model_name):
        for i, network in enumerate(self.model):
            tl.files.save_npz(network.all_params, 
                name='../model/ppo/{}_{}.npz'.format(model_name, i),
                sess=self.sess)

    def load_network(self, model_name):
        for i, network in enumerate(self.model):
            params = tl.files.load_npz(
                name='../model/ppo/{}_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, network)







