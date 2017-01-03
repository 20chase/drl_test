# author: ftx
# data: 2017/1/1
# e-mail: ftx1994@foxmail.com

# for deep reinforcement learning
import tensorflow as tf
import tensorlayer as tl
import gym
# for numerical process
import numpy as np
# for system process
import time
import os
from collections import deque

# hyber parameters
GAMMA = 1
LEARNING_RATE = 5e-3
BATCH_SIZE = 10
REPLAY_MEMORY = 1e5
TEST = 100
TIME_OUT = 30
SUPERVISED = False
LOAD = False
SAVE = False
DISPLAY = False
MODE_NAME = 'CartPole-v0'

class Brain_PG():
    def __init__(self, env):

        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        self.creat_network()

        self.train()

        self.session = tf.InteractiveSession()

        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

        self.session.run(tf.global_variables_initializer())

        self.replay_memory = deque()

    def creat_network(self):
        self.state_batch = tf.placeholder(tf.float32, shape=[None, self.state_dim])

        self.network = tl.layers.InputLayer(self.state_batch)
        self.network = tl.layers.DenseLayer(self.network, n_units=60, act=tf.nn.relu, name='relu1')
        self.network = tl.layers.DenseLayer(self.network, n_units=60, act=tf.nn.relu, name='relu2')
        self.network = tl.layers.DenseLayer(self.network, n_units=60, act=tf.nn.relu, name='relu3')
        self.network = tl.layers.DenseLayer(self.network, n_units=self.action_dim)

        self.output = self.network.outputs

        self.output_prob = tf.nn.softmax(self.output)


    def train(self):
        self.action_batch = tf.placeholder(tf.int32, shape=[None])

        self.reward_batch = tf.placeholder(tf.float32, shape=[None])

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.action_batch)
        self.loss = tf.reduce_sum(tf.mul(self.cross_entropy, self.reward_batch))   

        self.reward_mean = tf.placeholder(tf.float32)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('reward'):
            tf.summary.scalar('reward_means', self.reward_mean)

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    # TODO: make a replay memory
    def perceive(self, state, action, reward, terminate, reward_sum):
        if terminate is True:
            reward = reward_sum
        else:
            reward = 0

        self.replay_memory.append((state, action, reward))
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()

    def action_softmax(self, state):
        state = state.reshape(1, self.state_dim)
        prob = self.output_prob.eval(feed_dict={self.state_batch: state})
        return np.random.choice(range(self.action_dim), p=prob.flatten())

    def action_deterministic(self, state):
    	state = state.reshape(1, self.state_dim)
    	return np.argmax(self.output.eval(feed_dict={self.state_batch: state}))

def discount_rewards(rewards = []):
    rewards = np.asarray(rewards)
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if rewards[t] != 0: running_add = 0

        running_add = running_add * GAMMA + rewards[t]
        discounted_r[t] = running_add

    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    return discounted_r

def main():
    env = gym.make(MODE_NAME)
    agent = Brain_PG(env)

    if LOAD is True:
        params = tl.files.load_npz(name=MODE_NAME + '.npz')
        tl.files.assign_params(agent.session, params, agent.network)

    state_batch, action_batch, reward_batch = [], [], []

    state = env.reset()

    episode = 0
    test_flag = False
    reward_sum = 0
    reward_mean = 0
    start_time = time.time()

    while True:
        
        if episode % TEST == 0:
            test_flag = True

        if DISPLAY == True or test_flag == True:
            env.render()

        if SUPERVISED is True:
            action_flag = raw_input("input action:")
            if action_flag == 'a':
                action = 1
            else:
                if action_flag == 's':
                    action = 2
                else:
                    action = 3

        if test_flag is True:
            action = agent.action_deterministic(state)
        else:
            action = agent.action_softmax(state)

        next_state, reward, terminate, _ = env.step(action)
        reward_sum += reward

        state_batch.append(state)
        action_batch.append(action)


        if time.time() - start_time > TIME_OUT:
        	terminate = True
        	print 'WARNMING: TIME OUT!'

        state = next_state

        if terminate is True:

            reward_batch.append(reward_sum)

            episode += 1

            print 'episode: ', episode, '    reward_sum: ', reward_sum
            reward_mean += reward_sum

            if episode % BATCH_SIZE == 0:
                print ' ...... update parameters ...... '
                state_batch = np.vstack(state_batch)
                action_batch = np.asarray(action_batch)
                reward_batch = discount_rewards(reward_batch)
                
                reward_mean = reward_mean / BATCH_SIZE

                print 'reward_mean = ', reward_mean

                summary, _ = agent.session.run([agent.merged, agent.optimizer], feed_dict = {agent.state_batch: state_batch, agent.action_batch: action_batch,agent.reward_batch: reward_batch, agent.reward_mean: reward_mean})

                agent.train_writer.add_summary(summary, episode)

                if SAVE is True:
                    tl.files.save_npz(agent.network.all_params, name=MODE_NAME + '.npz')

                state_batch, action_batch, reward_batch = [], [], []
                reward_mean = 0

            state = env.reset()
            test_flag = False
            reward_sum = 0
            start_time = time.time()

        else:
            reward_batch.append(0)

    

if __name__ == '__main__':
    os.popen('rm -rf /tmp/train')
    main()