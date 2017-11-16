import gym
import random

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from gym import wrappers
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.5 # final value of epsilon
EXPLOER_NUM = 10000
REPLAY_SIZE = 20000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
LEARNING_RATE = 5e-4
DECLAY_RATE = 1e-4
DECLAY_FLAG = True
DISPLAY = False
SAVE = False
LOAD = False
# MODE_NAME = 'LunarLander-v2'
MODE_NAME = 'CartPole-v1'
EPISODE = 10000 # Episode limitation
STEP = 10000 # Step limitation in an episode
TEST = 50

UPDATE_TIME = 500
OBSERVE_NUM = 64
TARGET_NUM = 995
EVAL_FLAG = False

class DDQN():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.reward = 0
        self.epsilon = INITIAL_EPSILON
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        
        self._build_ph()
        self.Q_value, self.network = self.create_Q_network('eval')
        self.Q_value_target, self.network_target = self.create_Q_network('target')
        # self.create_Q_network_target()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/A', self.session.graph)
        self.session.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], name='obs_ph')
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], name='act_ph') # one hot presentation
        self.ret_ph = tf.placeholder(tf.float32, [None], name='ret_ph')

    def create_Q_network(self, model_name):
        network = tl.layers.InputLayer(self.obs_ph, name='input_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=64, act=tf.nn.relu, name='relu1_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=32, act=tf.nn.relu, name='relu2_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=16, act=tf.nn.relu, name='relu3_{}'.format(model_name))
        network = tl.layers.DenseLayer(network, n_units=self.act_dim, name='output_{}'.format(model_name))

        q_value = network.outputs
        return q_value, network

    def create_training_method(self):
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.act_ph), axis=1)
        self.cost = tf.reduce_mean(tf.square(self.ret_ph - Q_action))
        self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def perceive(self, state, action,reward,next_state,done):
        one_hot_action = np.zeros(self.act_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def write_reward(self, reward_sum):
        self.reward = reward_sum

    def train_Q_network(self):
        self.time_step += 1
        
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Q_value_target_batch = self.Q_value_target.eval(feed_dict={self.obs_ph:next_state_batch})
        Q_value_batch = self.Q_value.eval(feed_dict={self.obs_ph:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * Q_value_target_batch[i][np.argmax(Q_value_batch[i])])

        replay_size = len(self.replay_buffer)

        feed_dict = {
        self.ret_ph:y_batch,
        self.act_ph:action_batch,
        self.obs_ph:state_batch
        }

        self.session.run(self.opt, feed_dict=feed_dict)

    def update_target(self):
        tl.files.assign_params(self.session, self.network.all_params, self.network_target)

    def egreedy_action(self,state):
        if self.time_step < OBSERVE_NUM:
            return random.randint(0,self.act_dim - 1)
        if DECLAY_FLAG:
            self.epsilon *= (1 - DECLAY_RATE)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLOER_NUM

        if self.epsilon < FINAL_EPSILON:
            self.epsilon *= (1 - DECLAY_RATE * 0.01)

        Q_value = self.Q_value.eval(feed_dict = {self.obs_ph:[state]})[0]

        if random.random() <= self.epsilon:
            return random.randint(0,self.act_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
        self.obs_ph:[state]
        })[0])


def train_game():  
    env = gym.make(MODE_NAME)
    agent = DDQN(env)

    if LOAD is True:
        params = tl.files.load_npz(name=MODE_NAME + '.npz')
        tl.files.assign_params(agent.session, params, agent.network)

    reward_mean = 0
    reward_sum = 0
    end_flag = False
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        if end_flag:
            break;

        for step in range(STEP):
            if DISPLAY is True:
                env.render()
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state,reward,done,_ = env.step(action)
            reward_sum += reward
            agent.perceive(state,action,reward,next_state,done)
            state = next_state

            if done:
                agent.write_reward(reward_sum)
                reward_mean += reward_sum
                agent.update_target()
                reward_sum = 0
                break

        if episode % TEST == 0:
            if SAVE is True:
                tl.files.save_npz(agent.network.all_params, name=MODE_NAME + '.npz')
            reward_mean /= (TEST + 1)

            if (reward_mean > TARGET_NUM):
                end_flag = True

            print ('episode:', episode, '   reward_mean:', reward_mean, '    epsilon: ', agent.epsilon)
    

if __name__ == '__main__':
    train_game()
    # if EVAL_FLAG:
    #     gym.upload('/tmp/' + MODE_NAME, api_key='sk_nXYWtyR0CfjmTgSiJVJA')
 