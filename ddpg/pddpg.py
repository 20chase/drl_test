import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils as U


class PrioritizedDDPG(object):
    def __init__(self, sess, args, obs_dim, act_dim):
        self.sess = sess
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

        self.ou_noise = U.OUNoise(self.act_dim)

        self.time_step = 0
        self.score = 0

        self._build_ph()
        self.eval_model, self.act_eval, self.q_eval = self._build_net('eval')
        self.target_model, self.act_target, self.q_target = self._build_net('target')
        self.mean_q, self.td_error, self.critic_loss, self.critic_opt, self.actor_opt, self.update_target = self._build_training_method()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tensorboard/ddpg/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)


    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')
        self.weights_ph = tf.placeholder(tf.float32, [None, ], 'weights_ph')
        self.q_grads_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'q_grads_ph')


    def _build_net(self, model_name):
        actor_net = tl.layers.InputLayer(self.obs_ph,
            name='{}_actor_input'.format(model_name))
        actor_net = tl.layers.DenseLayer(actor_net, n_units=400, act=tf.nn.relu,
            name='{}_actor_relu1'.format(model_name))
        actor_net = tl.layers.DenseLayer(actor_net, n_units=300, act=tf.nn.relu,
            name='{}_actor_relu2'.format(model_name))
        actor_net = tl.layers.DenseLayer(actor_net, n_units=self.act_dim, act=tf.nn.tanh,
            name='{}_actor_output'.format(model_name))

        action = actor_net.outputs

        critic_net = tl.layers.InputLayer(self.obs_ph,
            name='{}_critic_input'.format(model_name))
        critic_net = tl.layers.DenseLayer(critic_net, n_units=400, act=tf.nn.relu,
            name='{}_critic_relu1'.format(model_name))
        critic_net = tl.layers.DenseLayer(critic_net, n_units=300,
            name='{}_critic_relu2'.format(model_name))

        act_net = tl.layers.InputLayer(self.act_ph,
            name='{}_act_input'.format(model_name))
        act_net = tl.layers.DenseLayer(act_net, n_units=300,
            name='{}_act_net'.format(model_name))

        q_net = tl.layers.InputLayer(act_net.outputs+critic_net.outputs,
            name='{}_q_input'.format(model_name))
        q_net = tl.layers.DenseLayer(q_net, n_units=300, act=tf.nn.relu,
            name='{}_q_relu'.format(model_name))
        q_net = tl.layers.DenseLayer(q_net, n_units=1,
            name='{}_q_output'.format(model_name))

        q_value = q_net.outputs

        return [actor_net, critic_net, act_net, q_net], action, q_value

    def _build_training_method(self):
        mean_q = tf.reduce_mean(self.q_eval)
        self.q_gradients = tf.gradients(self.q_eval, self.act_ph)
        with tf.variable_scope('td_error'):
            td_error = tf.squeeze(self.q_eval) - self.ret_ph 
        if self.args.huber:
            with tf.variable_scope('huber_loss'):
                errors = U.huber_loss(td_error)
        else:
            with tf.variable_scope('mse_loss'):
                errors = tf.square(td_error)

        with tf.variable_scope('critic_loss'):
            critic_loss = tf.reduce_mean(self.weights_ph*errors)

        critic_opt = tf.train.AdamOptimizer(self.args.critic_lr).minimize(critic_loss)

        actor_net = self.eval_model[0]
        
        grads = tf.gradients(self.act_eval, actor_net.all_params, -self.q_grads_ph)

        actor_opt = tf.train.AdamOptimizer(self.args.actor_lr).apply_gradients(zip(grads, actor_net.all_params))

        # update target network
        eval_vars, target_vars = [], []

        for eval_net, target_net in zip(self.eval_model, self.target_model):
            eval_vars += eval_net.all_params
            target_vars += target_net.all_params

        update_target = []
        for var, var_target in zip(eval_vars, target_vars):
            update_target.append(var_target.assign(var))

        update_target = tf.group(*update_target)

        return mean_q, td_error, critic_loss, critic_opt, actor_opt, update_target

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        self.size_tb = tf.placeholder(tf.float32, name='size_tb')
        self.beta_tb = tf.placeholder(tf.float32, name='beta_tb')

        with tf.name_scope('loss'):
            tf.summary.scalar('critic_loss', self.critic_loss)

        with tf.name_scope('params'):
            tf.summary.scalar('q-value', self.mean_q)
            tf.summary.scalar('score', self.score_tb)
            tf.summary.scalar('buffer_size', self.size_tb)
            if self.args.prioritized:
                tf.summary.scalar('beta', self.beta_tb)

    def train(self):
        self.time_step += 1
        if self.args.prioritized:
            self.beta += self.speed_beta
            experience = self.buffer.sample(self.args.batch_size, self.beta)
            (obses, acts, rews, new_obses, dones, weights, idxes) = experience
        else:
            obses, acts, rews, new_obses, dones = self.buffer.sample(self.args.batch_size)
            weights, idxes = np.ones_like(rews), None

        act_target = self.sess.run(self.act_target, feed_dict={self.obs_ph: new_obses})
        q_target = self.sess.run(self.q_target, feed_dict={self.obs_ph: new_obses, self.act_ph: act_target})

        rets = []
        for i in range(self.args.batch_size):
            if dones[i]:
                rets.append(rews[i])
            else:
                rets.append(rews[i]+self.args.gamma*q_target[i][0])

        rets = np.asarray(rets)

        feed_dict = {
        self.obs_ph: obses,
        self.act_ph: acts,
        self.ret_ph: rets,
        self.weights_ph: weights,
        self.score_tb: self.score,
        self.size_tb: len(self.buffer)
        }

        if self.args.prioritized:
            feed_dict[self.beta_tb] = self.beta

        summary, td_error, _ = self.sess.run([self.merge_all, self.td_error, self.critic_opt],
            feed_dict=feed_dict)

        # write tensorboard file
        self.writer.add_summary(summary, self.time_step)

        act_eval = self.sess.run(self.act_eval, feed_dict={self.obs_ph: obses})
        grads = self.sess.run(self.q_gradients, feed_dict={self.obs_ph: obses, self.act_ph: act_eval})[0]

        self.sess.run(self.actor_opt, feed_dict={self.obs_ph: obses, self.q_grads_ph: grads})

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
        action = self.sess.run(self.act_eval, feed_dict=feed_dict)[0]
        if self.args.test_alg or test:
            return action
        else:
            return (action + self.ou_noise.noise())

    def get_score(self, score):
        self.score = score

    def save_network(self, model_name):
        for i, network in enumerate(self.eval_model):
            tl.files.save_npz(network.all_params, 
                name='../model/ddpg/{}_{}.npz'.format(model_name, i),
                sess=self.sess)

    def load_network(self, model_name):
        for i, network in enumerate(self.eval_model):
            params = tl.files.load_npz(
                name='../model/ddpg/{}_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, network)






