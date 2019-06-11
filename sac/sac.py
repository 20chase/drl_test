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


class SAC(object):
    def __init__(self, sess, args, 
                 obs_dim, act_dim,
                 alpha=0.2):

        self.sess = sess
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alpha = 0.2

        self.tau = 0.995

        self.time_step = 0
        self.counter = 0
        self.score = 0

        self._build_ph()
        with tf.variable_scope("main"):
            self.mu, self.pi, self.logp_pi, \
            self.q1, self.q2, self.q1_pi, self.q2_pi, \
            self.v = self._build_net(self.obs_ph, self.act_ph)

        with tf.variable_scope("target"):
            _, _, _, _, _, _, _, \
            self.v_targ = self._build_net(self.new_obs_ph, self.act_ph)

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=args.replay_size)


        self._build_training_method()

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

    def _build_net(self, x, a, 
                   hidden_sizes=(400,300), 
                   activation=tf.nn.relu, 
                   output_activation=None):

        def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
            for h in hidden_sizes[:-1]:
                x = tf.layers.dense(x, units=h, activation=activation)
            return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

        def gaussian_likelihood(x, mu, log_std):
            EPS = 1e-8
            pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
            return tf.reduce_sum(pre_sum, axis=1)

        def clip_but_pass_gradient(x, l=-1., u=1.):
            clip_up = tf.cast(x > u, tf.float32)
            clip_low = tf.cast(x < l, tf.float32)
            return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

        def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
            LOG_STD_MAX = 2
            LOG_STD_MIN = -20
            
            act_dim = a.shape.as_list()[-1]
            net = mlp(x, list(hidden_sizes), activation, activation)
            mu = tf.layers.dense(net, act_dim, activation=output_activation)

            """
            Because algorithm maximizes trade-off of reward and entropy,
            entropy must be unique to state---and therefore log_stds need
            to be a neural network output instead of a shared-across-states
            learnable parameter vector. But for deep Relu and other nets,
            simply sticking an activationless dense layer at the end would
            be quite bad---at the beginning of training, a randomly initialized
            net could produce extremely large values for the log_stds, which
            would result in some actions being either entirely deterministic
            or too random to come back to earth. Either of these introduces
            numerical instability which could break the algorithm. To 
            protect against that, we'll constrain the output range of the 
            log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
            slightly different from the trick used by the original authors of
            SAC---they used tf.clip_by_value instead of squashing and rescaling.
            I prefer this approach because it allows gradient propagation
            through log_std where clipping wouldn't, but I don't know if
            it makes much of a difference.
            """
            log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi = gaussian_likelihood(pi, mu, log_std)
            return mu, pi, logp_pi

        def apply_squashing_func(mu, pi, logp_pi):
            mu = tf.tanh(mu)
            pi = tf.tanh(pi)
            # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
            logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
            return mu, pi, logp_pi

        with tf.variable_scope("pi"):
            mu, pi, logp_pi = mlp_gaussian_policy(
                x, a, hidden_sizes, activation, output_activation)
            mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
            
        # vfs
        vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([x,a], axis=-1))
        with tf.variable_scope('q1', reuse=True):
            q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([x,a], axis=-1))
        with tf.variable_scope('q2', reuse=True):
            q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
        with tf.variable_scope('v'):
            v = vf_mlp(x)
        return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

    def _build_training_method(self):
        def get_vars(scope):
            return [x for x in tf.global_variables() if scope in x.name]

        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

        q_backup = tf.stop_gradient(
            self.rew_ph + \
            self.args.gamma*(1-self.done_ph)*self.v_targ)

        v_backup = tf.stop_gradient(
            min_q_pi - self.alpha * self.logp_pi)

        pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        value_loss = q1_loss + q2_loss + v_loss

        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        value_params = get_vars('main/q') + get_vars('main/v')

        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, self.tau*v_targ + (1-self.tau)*v_main)
                                    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [pi_loss, q1_loss, q2_loss, v_loss,
                    self.q1, self.q2, self.v, self.logp_pi, 
                    train_pi_op, train_value_op, target_update]

        self.target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

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
        batch = self.replay_buffer.sample_batch(self.args.batch_size)

        feed_dict = {
        self.obs_ph: batch['obs1'],
        self.new_obs_ph: batch['obs2'],
        self.act_ph: batch['acts'],
        self.rew_ph: batch['rews'],
        self.done_ph: batch['done']
        }

        outs = self.sess.run(self.step_ops, feed_dict)

        return outs

    def action(self, obs, test=False):
        feed_dict =  {self.obs_ph: obs}
        act_op = self.mu if test else self.pi
        return self.sess.run(act_op, feed_dict=feed_dict)

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






