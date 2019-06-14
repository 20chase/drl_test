import time
import joblib

import numpy as np
import tensorflow as tf
import utils as U

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class TD3(object):
    def __init__(self, sess, args, obs_dim, act_dim):
        self.sess = sess
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.theta = args.theta
        self.sigma = args.sigma

        self.tau = 0.995

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

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=args.replay_size)

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

    def perceive(self, obs, acts, rews, new_obs, dones):
        if self.args.train:
            for i in range(len(dones)):
                if dones[i]:
                    done = 1.0
                else:
                    done = 0.0

                self.replay_buffer.store(
                    obs[i], acts[i], rews[i], new_obs[i], done
                )
                
    def train(self):
        self.time_step += 1

        batch = self.replay_buffer.sample_batch(self.args.batch_size)

        feed_dict = {
        self.obs_ph: batch['obs1'],
        self.new_obs_ph: batch['obs2'],
        self.act_ph: batch['acts'],
        self.rew_ph: batch['rews'],
        self.done_ph: batch['done']
        }

        self.sess.run(
            self.critic_opt,
            feed_dict=feed_dict)

        self.sess.run(
            [self.actor_opt, self.target_update], 
            feed_dict=feed_dict)

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






