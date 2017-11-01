#! /usr/bin/env python3
import argparse
import gym
import time
import roboschool
import scipy.signal

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tabulate import tabulate
from gym import wrappers
from collections import OrderedDict
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='cliped ppo algorithm')

parser.add_argument(
    '--gamma', default=0.995, type=float, help='gamma')

parser.add_argument(
    '--lambda_gae', default=0.98, type=float, help='lambda for GAE')

parser.add_argument(
    '--epsilon', default=0.2, type=float, help='cliped parameter')

parser.add_argument(
    '--lr', default=3e-4, type=float, help='learning rate')

parser.add_argument(
    '--kl_targ', default=0.01, type=float, help='kl divergence target')

parser.add_argument(
    '--train_epochs', default=10, type=int, help='training epochs')

parser.add_argument(
    '--batch_size', default=20, type=int, help='trianing batch size')

parser.add_argument(
    '--training_steps', default=4000, type=int, help='steps number for training')

parser.add_argument(
    '--max_episodes', default=1000000000, type=int, help='max trianing episodes')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--save_network', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load_network', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--test_algorithm', default=False, type=bool, help='wether to test algorithm')

parser.add_argument(
    '--eval_algorithm', default=False, type=bool, help='whether to evaluate algorithm')

parser.add_argument(
    '--env_name', default='RoboschoolInvertedPendulum-v1', type=str, help='gym env name')

parser.add_argument(
    '--model_name', default='ppo_clip', type=str, help='save or load model name')

args = parser.parse_args()


class PPO(object):
    def __init__(self, env, args):
        self.init_param(env, args)
        self.session = tf.InteractiveSession()

        self._build_ph()
        self.model = self._build_network()
        self._build_training()
        self._build_summary()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./tensorboard/ppo/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            ), self.session.graph)
        self.session.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim],
                                     'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim],
                                     'act_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ],
                                     'adv_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ],
                                     'ret_ph')

        self.lr_ph = tf.placeholder(tf.float32, name='lr_ph')

        self.epsilon_ph = tf.placeholder(tf.float32, name='epsilon_ph')

        self.old_log_vars_ph = tf.placeholder(tf.float32, [self.act_dim, ],
                                              'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, [None, self.act_dim],
                                           'old_means')

    def _build_network(self):
        # define network
        self.network = tl.layers.InputLayer(self.obs_ph, 
                                            name='network_input')
        self.network = tl.layers.DenseLayer(self.network,
                                            n_units=100,
                                            act=tf.nn.relu,
                                            W_init=tf.random_normal_initializer(stddev=np.sqrt(1.0/self.obs_dim)),
                                            name='actor_relu1')
        self.network = tl.layers.DenseLayer(self.network,
                                            n_units=50,
                                            act=tf.nn.relu,
                                            W_init=tf.random_normal_initializer(stddev=np.sqrt(1.0/float(100))),
                                            name='actor_relu2')
        self.network = tl.layers.DenseLayer(self.network,
                                            n_units=25,
                                            act=tf.nn.relu,
                                            W_init=tf.random_normal_initializer(stddev=np.sqrt(1.0/float(50))),
                                            name='actor_relu3')
        self.network = tl.layers.DenseLayer(self.network,
                                            n_units=self.act_dim,
                                            act=tf.nn.tanh,
                                            W_init=tf.random_normal_initializer(stddev=np.sqrt(1.0/float(25))),
                                            name='means')

        self.value_network = tl.layers.InputLayer(self.obs_ph,
                                                  name='value_network_input')
        self.value_network = tl.layers.DenseLayer(self.value_network,
                                                  n_units=100,
                                                  act=tf.nn.relu,
                                                  name='critic_relu1'
                                                  )
        self.value_network = tl.layers.DenseLayer(self.value_network,
                                                  n_units=50,
                                                  act=tf.nn.relu,
                                                  name='critic_relu2'
                                                  )
        self.value_network = tl.layers.DenseLayer(self.value_network,
                                                  n_units=25,
                                                  act=tf.nn.relu,
                                                  name='critic_relu3'
                                                  )
        self.value_network = tl.layers.DenseLayer(self.value_network,
                                                  n_units=1,
                                                  name='value_output')
        

        logvar_speed = 30
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim),
                                   tf.float32, tf.constant_initializer(0.0))

        self.log_vars = tf.reduce_sum(log_vars, axis=0)-1.
        self.means = self.network.outputs
        self.value = self.value_network.outputs

        with tf.variable_scope('sample_action'):
            self.sampled_act = (self.means +
                                tf.exp(self.log_vars/2.0) *
                                tf.random_normal(shape=(self.act_dim,)))

        return [self.network, self.value_network]

    def _build_training(self):
        # logprob
        self.logp = -0.5*tf.reduce_sum(self.log_vars) + -0.5*tf.reduce_sum(tf.square(self.act_ph-self.means) /
                           tf.exp(self.log_vars), axis=1)
        self.logp_old = -0.5*tf.reduce_sum(self.old_log_vars_ph) + -0.5*tf.reduce_sum(tf.square(self.act_ph-self.old_means_ph) /
                           tf.exp(self.old_log_vars_ph), axis=1)

        with tf.variable_scope('kl'):
            self.kl = 0.5*tf.reduce_mean(tf.reduce_sum(tf.exp(self.old_log_vars_ph-self.log_vars)) + 
                                         tf.reduce_sum(tf.square(self.means-self.old_means_ph)/tf.exp(self.log_vars), axis=1) -
                                         self.act_dim +
                                         tf.reduce_sum(self.log_vars) - tf.reduce_sum(self.old_log_vars_ph))

        with tf.variable_scope('entropy'):
            self.entropy = 0.5*(self.act_dim*(np.log(2*np.pi)+1) +
                                tf.reduce_sum(self.log_vars))

        with tf.variable_scope('surogate_loss'):
            ratio = tf.exp(self.logp-self.logp_old)
            surr1 = ratio*self.adv_ph
            surr2 = tf.clip_by_value(ratio, 1.-self.epsilon_ph, 1.+self.epsilon_ph) * self.adv_ph
            self.surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        with tf.variable_scope('value_loss'):
            self.value_loss = tf.reduce_mean(tf.square(self.value-self.ret_ph))

        with tf.variable_scope('total_loss'):
            self.total_loss = self.surr_loss + 1.0 * self.value_loss + 0.0 * self.entropy

        # vf_grad = tf.gradients(self.value_loss, self.critic_network.all_params)

        self.actor_opt = tf.train.AdamOptimizer(self.lr_ph).minimize(self.total_loss)
        self.critic_opt = tf.train.AdamOptimizer(1e-3).minimize(self.value_loss)
               
    def _build_summary(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        self.lr_tb = tf.placeholder(tf.float32, name='lr_tb')
        self.epsilon_tb = tf.placeholder(tf.float32, name='epsilon_tb')
        self.kl_tb = tf.placeholder(tf.float32, name='kl_tb')
        self.entropy_tb = tf.placeholder(tf.float32, name='entropy_tb')
        self.surr_loss_tb = tf.placeholder(tf.float32, name='surr_loss_tb')
        self.value_loss_tb = tf.placeholder(tf.float32, name='value_loss_tb')
        self.total_loss_tb = tf.placeholder(tf.float32, name='total_loss_tb')

        with tf.name_scope('loss'):
            tf.summary.scalar('surr_loss', self.surr_loss_tb)
            tf.summary.scalar('value_loss', self.value_loss_tb)
            tf.summary.scalar('total_loss', self.total_loss_tb)

        with tf.name_scope('param'):
            tf.summary.scalar('entropy', self.entropy_tb)
            tf.summary.scalar('kl', self.kl_tb)
            tf.summary.scalar('lr', self.lr_tb)
            tf.summary.scalar('epsilon', self.epsilon_tb)

        tf.summary.scalar('score', self.score_tb)

    def update_actor(self, obs, acts, advs, rets, score):
        feed_dict = {
        self.obs_ph: obs,
        self.act_ph: acts,
        self.adv_ph: advs,
        self.ret_ph: rets,
        self.lr_ph: self.args.lr * self.lr_multiplier,
        self.epsilon_ph: self.args.epsilon * self.lr_multiplier
        }

        old_means_np, old_log_vars_np = self.session.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np

        if not self.args.test_algorithm:
            for e in range(self.args.train_epochs):
                self.session.run(self.actor_opt, feed_dict)
                kl = self.session.run(self.kl, feed_dict)
                if kl > self.args.kl_targ * 4:
                    break

        if kl > self.args.kl_targ * 2.:
            self.lr_multiplier /= 1.5
        elif kl < self.args.kl_targ / 2.:
            self.lr_multiplier *= 1.5

        stats = self._visualize_stats(feed_dict, score)
        self._visualize_tensorboard(stats)

        if self.args.save_network:
            self.save_network(self.args.model_name)

        return stats

    def _visualize_stats(self, feed_dict, score):
        kl, entropy, surr_loss, value_loss, total_loss = self.session.run(
            [self.kl, self.entropy, self.surr_loss, self.value_loss, self.total_loss], 
             feed_dict)

        stats = OrderedDict()
        stats["Score"] = score
        stats["LearningRate"] = self.args.lr * self.lr_multiplier
        stats["ClipedParameter"] = self.args.epsilon * self.lr_multiplier
        stats["KL-divergence"] = kl
        stats["Entropy"] = entropy
        stats["SurrogateLoss"] = surr_loss
        stats["ValueLoss"] = value_loss
        stats["Loss"] = total_loss

        return stats

    def _visualize_tensorboard(self, stats):
        feed_dict = {
        self.score_tb: stats["Score"],
        self.lr_tb: stats["LearningRate"],
        self.epsilon_tb: stats["ClipedParameter"],
        self.kl_tb: stats["KL-divergence"],
        self.entropy_tb: stats["Entropy"],
        self.surr_loss_tb: stats["SurrogateLoss"],
        self.value_loss_tb: stats["ValueLoss"],
        self.total_loss_tb: stats["Loss"]
        }

        self.time_step += 1
        summary = self.session.run(self.merge_all, feed_dict)
        self.writer.add_summary(summary, self.time_step)

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

    def save_network(self, model_name):
        for i in range(len(self.model)):
            tl.files.save_npz(self.model[i].all_params,
                              name='./model/ppo/{}_{}.npz'.format(model_name, i),
                              sess=self.session)

    def load_network(self, model_name):
        for i in range(len(self.model)):
            params = tl.files.load_npz(name='./model/ppo/{}_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.session, params, self.model[i])

    def sample(self, obs):
        obs = np.reshape(obs, (1, self.obs_dim))
        feed_dict = {self.obs_ph: obs}
        if self.args.test_algorithm:
            return self.session.run(self.means, feed_dict=feed_dict)
        else:
            return self.session.run(self.sampled_act, feed_dict=feed_dict)

    def get_value(self, obs):
        return self.value.eval(feed_dict={self.obs_ph: obs})

    def convert_action(self, action):
        return action*self.act_high

    def init_param(self, env, args):
        self.args = args
        # env param
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_high = env.action_space.high

        # value init
        self.time_step = 0
        self.score = 0
        self.lr_multiplier = 1.
        self.critic_epochs = 10
        self.replay_buffer_x = None
        self.replay_buffer_y = None

def run_episode(env, agent, animate=args.animate):
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

def run_policy(env, agent, training_steps, batch_size):
    trajectories = []
    total_step = 0
    for e in range(batch_size):
        obs, acts, rewards = run_episode(env, agent)
        acts = np.reshape(acts, (len(rewards), agent.act_dim))
        total_step += len(rewards)
        trajectory = {
        'obs': obs,
        'acts': acts,
        'rewards': rewards
        }
        trajectories.append(trajectory)

    score = np.mean([t['rewards'].sum() for t in trajectories])
    return trajectories, score, total_step

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

def print_stats(stats):
    print("*********** Iteration {} ************".format(stats["Iteration"]))
    table = []
    for k, v in stats.items():
        table.append([k, v])

    print(tabulate(table, tablefmt="grid"))

def train():
    env = gym.make(args.env_name)
    agent = PPO(env, args)
    if args.eval_algorithm:
        env = wrappers.Monitor(env, './model/{}'.format(args.model_name), force=True)
    e = 0

    if args.load_network:
        agent.load_network(args.model_name)
    while e < (args.max_episodes):
        trajectories, score, total_step = run_policy(env, agent, args.training_steps, args.batch_size)
        e += len(trajectories)
        add_value(trajectories, agent)
        add_disc_sum_rew(trajectories, args.gamma)
        add_gae(trajectories, args.gamma, args.lambda_gae)
        obs, acts, advs, rets = build_train_set(trajectories)

        stats = agent.update_actor(obs, acts, advs, rets, score)
        # agent.update_critic(obs, rets)
        stats["AverageStep"] = total_step / args.batch_size
        stats["Iteration"] = e
        print_stats(stats)

if __name__ == "__main__":
    train()
