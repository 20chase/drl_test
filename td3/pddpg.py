import time
import joblib

import numpy as np
import tensorflow as tf
import utils as U


class PrioritizedDDPG(object):
    def __init__(self, sess, args, obs_dim, act_dim):
        self.sess = sess
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.theta = args.theta
        self.sigma = args.sigma

        self.tau = 0.995

        self.buffer = U.PrioritizedReplayBuffer(
            self.args.buffer_size, alpha=self.args.alpha)
        self.speed_beta = (1. - self.args.beta) / self.args.max_steps
        self.beta = self.args.beta

        self.time_step = 0
        self.counter = 0
        self.score = 0

        self._build_ph()
        with tf.variable_scope("main"):
            self.act_eval, self.q1, self.q2, self.q1_pi = self._build_net(self.obs_ph, self.act_ph)
        with tf.variable_scope("target"):
            self.act_target, _, _, _ = self._build_net(self.new_obs_ph, self.act_ph)

        with tf.variable_scope("target", reuse=True):
            epsilon = tf.random_normal(tf.shape(self.act_target), stddev=0.2)
            epsilon = tf.clip_by_value(epsilon, -0.5, 0.5)
            a2 = self.act_target + epsilon
            _, self.q1_targ, self.q2_targ, _ = self._build_net(self.new_obs_ph, a2)

        self.td_error, \
        self.critic_loss, self.critic_opt, self.actor_opt, \
        self.target_init, self.target_update = self._build_training_method()
        
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tensorboard/td3/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

    def _build_ph(self):
        self.obs_ph = tf.placeholder(
            tf.float32, 
            [None, self.obs_dim], 
            'obs_ph')
        self.new_obs_ph = tf.placeholder(
            tf.float32, 
            [None, self.obs_dim], 
            'obs_ph')
        self.act_ph = tf.placeholder(
            tf.float32, 
            [None, self.act_dim], 
            'act_ph')
        self.rew_ph = tf.placeholder(
            tf.float32, 
            [None, ], 
            'ret_ph')
        self.done_ph = tf.placeholder(
            tf.float32,
            [None, ],
            'done_ph')
        self.weights_ph = tf.placeholder(
            tf.float32, 
            [None, ], 
            'weights_ph')

    def _build_net(self, obs_ph, act_ph):
        with tf.variable_scope("pi"):
            x = tf.layers.dense(
                obs_ph, units=400, activation=tf.nn.relu
                )
            x = tf.layers.dense(
                x, units=300, activation=tf.nn.relu
                )
            means = tf.layers.dense(
                x, units=self.act_dim, activation=tf.nn.tanh
                )

        with tf.variable_scope("q1"):
            x = tf.layers.dense(
                tf.concat([obs_ph, act_ph], axis=1), units=400, 
                activation=tf.nn.relu
            )
            x = tf.layers.dense(
                x, units=300, activation=tf.nn.relu
            )
            q1 = tf.layers.dense(
                x, units=1
            )

        with tf.variable_scope("q2"):
            x = tf.layers.dense(
                tf.concat([obs_ph, act_ph], axis=1), units=400, 
                activation=tf.nn.relu
            )
            x = tf.layers.dense(
                x, units=300, activation=tf.nn.relu
            )
            q2 = tf.layers.dense(
                x, units=1
            )

        with tf.variable_scope("q1", reuse=True):
            x = tf.layers.dense(
                tf.concat([obs_ph, means], axis=1), units=400, 
                activation=tf.nn.relu
            )
            x = tf.layers.dense(
                x, units=300, activation=tf.nn.relu
            )
            q1_pi = tf.layers.dense(
                x, units=1
            )

        return means, \
               tf.squeeze(q1, axis=1), \
               tf.squeeze(q2, axis=1), \
               tf.squeeze(q1_pi, axis=1)

    def _build_training_method(self):
        min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)
        backup = tf.stop_gradient(
            self.rew_ph + self.args.gamma*(1-self.done_ph)*min_q_targ)
            
        with tf.variable_scope('td_error'):
            td_error = min_q_targ - backup
        
        with tf.variable_scope('q1_loss'):
            q1_loss = tf.reduce_mean(
                self.weights_ph * (self.q1 - backup) ** 2
            )

        with tf.variable_scope('q2_loss'):
            q2_loss = tf.reduce_mean(
                self.weights_ph * (self.q2 - backup) **2
            )

        with tf.variable_scope('q_loss'):
            q_loss = q1_loss + q2_loss

        with tf.variable_scope("actor_loss"):
            actor_loss = -tf.reduce_mean(self.q1_pi)

        critic_param = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="main/q"
        )

        actor_param = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="main/pi"
        )

        critic_opt = tf.train.AdamOptimizer(
            self.args.critic_lr).minimize(
                q_loss, var_list=critic_param)

        actor_opt = tf.train.AdamOptimizer(
            self.args.actor_lr).minimize(
                actor_loss, var_list=actor_param)

        # update target network
        eval_vars, target_vars = [], []

        eval_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="main"
        )

        eval_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="target"
        )


        target_init = tf.group(
            [tf.assign(v_targ, v_main)
            for v_main, v_targ in zip(eval_vars, target_vars)]
        )
        target_update = tf.group(
            [tf.assign(v_targ, self.tau*v_targ + (1-self.tau)*v_main)
            for v_main, v_targ in zip(eval_vars, target_vars)]
        )

        return td_error, \
               q_loss, critic_opt, actor_opt, \
               target_init, target_update

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(
            tf.float32, name='score_tb')
        self.size_tb = tf.placeholder(
            tf.float32, name='size_tb')
        self.beta_tb = tf.placeholder(
            tf.float32, name='beta_tb')

        with tf.name_scope('loss'):
            tf.summary.scalar('critic_loss', self.critic_loss)

        with tf.name_scope('params'):
            tf.summary.scalar('score', self.score_tb)
            tf.summary.scalar('buffer_size', self.size_tb)
            tf.summary.scalar('beta', self.beta_tb)

    def perceive(self, obs, acts, rews, new_obs, dones):
        self.counter += 1
        if self.args.train:
            for i in range(len(dones)):
                if dones[i]:
                    done = 1.0
                else:
                    done = 0.0

                self.buffer.add(
                    obs[i], acts[i], rews[i], new_obs[i], done
                )
                
    def train(self):
        self.time_step += 1
        for i in range(20):
            self.beta += self.speed_beta
            experience = self.buffer.sample(self.args.batch_size, self.beta)
            (obses, acts, rews, new_obses, dones, weights, idxes) = experience

            feed_dict = {
            self.obs_ph: obses,
            self.new_obs_ph: new_obses,
            self.act_ph: acts,
            self.rew_ph: rews,
            self.done_ph: dones,
            self.weights_ph: weights
            }

            td_error, _ = self.sess.run(
                [self.td_error, self.critic_opt],
                feed_dict=feed_dict)

            if i % 2 == 0:
                self.sess.run(
                    [self.actor_opt, self.target_update], 
                    feed_dict=feed_dict)

            new_priorities = np.abs(td_error) + 1e-6
            self.buffer.update_priorities(idxes, new_priorities)

    def ou_noise(self):
        return self.sigma * np.random.randn(self.args.nenvs, self.act_dim)
        
    def action(self, obs, test=False):
        feed_dict =  {self.obs_ph: obs}
        acts = self.sess.run(self.act_eval, feed_dict=feed_dict)
        if test:
            return acts
        else:
            return np.clip(acts + self.ou_noise(), -1.0, 1.0)

    def save_net(self, save_path):
        params = []
        params += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="main"
        )
        
        ps = self.sess.run(params)
        joblib.dump(ps, save_path)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = []
        params += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="main"
        )
        
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)






