#! /usr/bin/env python3
import ray
import gym

import numpy as np
import tensorflow as tf
import tensorlayer as tl


class Agent(object):
    def __init__(self, sess, obs_dim, act_dim, act_bound):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_high = act_bound

        self._build_ph()
        self.network, self.act = self._build_network()

        self.sess.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')

    def _build_network(self):
        network = tl.layers.InputLayer(self.obs_ph, name='input')
        network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.relu, name='hide1')
        network = tl.layers.DenseLayer(network, n_units=self.act_dim, act=tf.nn.tanh, name='out')
        outputs = network.outputs
        action = outputs * self.act_high
        return network, action

    def action(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.act, feed_dict)


# @ray.remote
# class RayEnvironment(object):
#     def __init__(self, env):
#         self.env = env
#         state = self.env.reset()
#         self.shape = state.shape

#     def step(self, action):
#         if self.done:
#             return [np.zeros(self.shape), 0.0, True]
#         else:
#             state, reward, done, info = self.env.step(action)
#             self.done = done
#             return [state, reward, done]

#     def reset(self):
#         self.done = False
#         return self.env.reset()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
        
@ray.remote
def run_episode(env, agent):
    env = env.x()
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.action([state])[0]
        state, reward, done, _ = env.step(action)
        score += reward
    return score
    
if __name__ == '__main__':
    num_cpus = 8
    ray.init(num_cpus=num_cpus)

    graph = tf.get_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True,
        intra_op_parallelism_threads=num_cpus,
        inter_op_parallelism_threads=num_cpus)

    sess = tf.Session(graph=graph, config=config)

    envs = [gym.make('Pendulum-v0') for _ in range(num_cpus)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_bound = envs[0].action_space.high

    agent = Agent(sess, obs_dim, act_dim, act_bound)
    
    remote_ids = [run_episode.remote(CloudpickleWrapper(env), agent) for env in envs]
    scores = ray.get(remote_ids)
    print (scores)



