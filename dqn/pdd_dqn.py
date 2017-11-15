import gym

import numpy as np
import tensorflow as tf
import tensorlayer as tl


class PDDDQN(object):
    def __init__(self, env):
        self.sess = tf.InteractiveSession()
        self._init_param(env)

    def _init_param(self, env):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.epsilon = 1.
        self.final_epsilon = 0.1

        self.gamma = 0.99
