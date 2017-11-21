import tensorflow as tf
import tensorlayer as tl
import gym

import numpy as np
import random
import os
from gym import wrappers
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EXPLOER_NUM = 10000
REPLAY_SIZE = 20000 # experience replay buffer size
BATCH_SIZE = 64 # size of minibatch
LEARNING_RATE = 1e-4
DECLAY_FLAG = True
DECLAY_NUM = 1e-5
DISPLAY = False
SAVE = False
LOAD = False
MODE_NAME = 'LunarLander-v2'
# MODE_NAME = 'Atlantis-ram-v0'
EPISODE = 10000 # Episode limitation
STEP = 10000 # Step limitation in an episode
TEST = 100

UPDATE_TIME = 500
OBSERVE_NUM = 32
TARGET_NUM = 195
EVAL_FLAG = False

class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.reward = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n
    print 'state_dim:', self.state_dim, '   action_dim:', self.action_dim

    self.create_Q_network()
    self.create_Q_network_target()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()

    self.merged = tf.summary.merge_all()

    self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

    self.session.run(tf.global_variables_initializer())

  def create_Q_network(self):
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    
    self.network = tl.layers.InputLayer(self.state_input, name='Input')
    self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu1')
    self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu2')
    self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu3')
    self.network = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='relu4')

    self.network_V = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='V_1')
    self.network_V = tl.layers.DenseLayer(self.network_V, n_units=200, act=tf.nn.relu, name='V_2')
    self.network_V = tl.layers.DenseLayer(self.network_V, n_units=1, name='output_V')

    self.network_A = tl.layers.DenseLayer(self.network, n_units=200, act=tf.nn.relu, name='A_1')
    self.network_A = tl.layers.DenseLayer(self.network_A, n_units=200, act=tf.nn.relu, name='A_2')
    self.network_A = tl.layers.DenseLayer(self.network_A, n_units=self.action_dim, name='output_A')
    
    
    self.value_function = self.network_V.outputs
    self.advantage_function = self.network_A.outputs

  def create_Q_network_target(self):
    # input layer
    self.state_input_target = tf.placeholder("float",[None,self.state_dim])
    
    self.network_target = tl.layers.InputLayer(self.state_input_target, name='Input_target')
    self.network_target = tl.layers.DenseLayer(self.network_target, n_units=200, act=tf.nn.relu, name='relu_target_1')
    self.network_target = tl.layers.DenseLayer(self.network_target, n_units=200, act=tf.nn.relu, name='relu_target_2')
    self.network_target = tl.layers.DenseLayer(self.network_target, n_units=200, act=tf.nn.relu, name='relu_target_3')
    self.network_target = tl.layers.DenseLayer(self.network_target, n_units=200, act=tf.nn.relu, name='relu_target_4')

    self.network_V_target = tl.layers.DenseLayer(self.network_target, n_units=200, act=tf.nn.relu, name='V_1_target')
    self.network_V_target = tl.layers.DenseLayer(self.network_V_target, n_units=200, act=tf.nn.relu, name='V_2_target')
    self.network_V_target = tl.layers.DenseLayer(self.network_V_target, n_units=1, name='output_V_target')

    self.network_A_target = tl.layers.DenseLayer(self.network_target, n_units=200, act=tf.nn.relu, name='A_1_target')
    self.network_A_target = tl.layers.DenseLayer(self.network_A_target, n_units=200, act=tf.nn.relu, name='A_2_target')
    self.network_A_target = tl.layers.DenseLayer(self.network_A_target, n_units=self.action_dim, name='output_A_target')

    self.value_function_target = self.network_V_target.outputs
    self.advantage_function_target = self.network_A_target.outputs
    
  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    self.reward_sum = tf.placeholder("float")
    self.epsilon_sum = tf.placeholder("float")
    self.replay_size = tf.placeholder("float")
    A_origin = tf.reduce_sum(tf.multiply(self.advantage_function, self.action_input),reduction_indices = 1)
    A_baseline = tf.reduce_mean(self.advantage_function, reduction_indices = 1)
    Q_action = self.value_function + (A_origin - A_baseline) 
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    
    A_value = tf.reduce_mean(self.advantage_function, axis=0)
    A_baseline_value = tf.reduce_mean(A_baseline)
    V_value = tf.reduce_mean(self.value_function)


    with tf.name_scope('loss'):
      tf.summary.scalar('cost', self.cost)

    with tf.name_scope('reward'):
      tf.summary.scalar('reward_mean', self.reward_sum)

    with tf.name_scope('Q_value_nomalize'):
		tf.summary.scalar('Q_value', V_value + (A_value[0] - A_baseline_value))
		tf.summary.scalar('Q_value', V_value + (A_value[1] - A_baseline_value))
		# tf.summary.scalar('Q_value', V_value + (A_value[2] - A_baseline_value))

    with tf.name_scope('param'):
		tf.summary.scalar('epsilon', self.epsilon_sum)
		tf.summary.scalar('replay_size', self.replay_size)

    self.optimizer_1 = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		# self.optimizer_2 = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-7).minimize(self.cost)
		# self.optimizer_3 = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-8).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
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
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []

    value_target_batch = self.value_function_target.eval(feed_dict = {self.state_input_target:next_state_batch})
    advantage_target_batch = self.advantage_function_target.eval(feed_dict = {self.state_input_target:next_state_batch})  
    advantage_baseline_batch = np.mean(advantage_target_batch, axis = 1)
    advantage_baseline_batch = advantage_baseline_batch.reshape(BATCH_SIZE, 1)
    advantage_batch = self.advantage_function.eval(feed_dict = {self.state_input:state_batch})

    # print '1:', np.shape(value_target_batch)
    # print '2:', np.shape(advantage_target_batch)
    # print '3:', np.shape(advantage_baseline_batch)
    # print '4:', np.shape(advantage_batch)

    # print '1-1:', value_target_batch[0][0]
    # print '3-1:', advantage_baseline_batch[0]
    # print '4-1:', np.argmax(advantage_batch[0])

   
    
    for i in range(0, BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * (value_target_batch[i][0] + (advantage_target_batch[i][np.argmax(advantage_batch[i])] - advantage_baseline_batch[i][0])))

    replay_size = len(self.replay_buffer)

    summary, _ = self.session.run([self.merged, self.optimizer_1], feed_dict={
    self.y_input:y_batch,
    self.action_input:action_batch,
    self.state_input:state_batch,
    self.reward_sum:self.reward,
    self.epsilon_sum:self.epsilon,
    self.replay_size:replay_size})


    self.train_writer.add_summary(summary, self.time_step)

    if self.time_step % UPDATE_TIME == 0:
      # print 'updating...'
      tl.files.assign_params(self.session, self.network.all_params, self.network_target)
      tl.files.assign_params(self.session, self.network_A.all_params, self.network_A_target)
      tl.files.assign_params(self.session, self.network_V.all_params, self.network_V_target)
		
      


    
  def egreedy_action(self,state):
    if self.time_step < OBSERVE_NUM:
      return random.randint(0,self.action_dim - 1)

    if DECLAY_FLAG:
      self.epsilon *= (1 - DECLAY_NUM)
    else:
      self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLOER_NUM

    if self.epsilon < FINAL_EPSILON:
      self.epsilon = FINAL_EPSILON

    Q_value = self.advantage_function.eval(feed_dict = {self.state_input:[state]})[0]

    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)
    else:
      return np.argmax(Q_value)

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])




def train_game():
  
  env = gym.make(MODE_NAME)
  if EVAL_FLAG:
    env = wrappers.Monitor(env, '/tmp/' + MODE_NAME)
  agent = DQN(env)

  if LOAD is True:
    params = tl.files.load_npz(name=MODE_NAME + '.npz')
    tl.files.assign_params(agent.session, params, agent.network)

  reward_mean = 0
  reward_sum = 0
  end_flag = False
  for episode in xrange(EPISODE):
    # initialize task
    state = env.reset()
    if end_flag:
      break;

    # Train
    for step in xrange(STEP):
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
        print 'epsido: ', episode, '... reward_sum: ', reward_sum
        reward_sum = 0
        if episode % TEST == 0:
          if SAVE is True:
            tl.files.save_npz(agent.network.all_params, name=MODE_NAME + '.npz')
          reward_mean /= (TEST + 1)

          if (reward_mean > TARGET_NUM):
            end_flag = True


          print 'episode:', episode, '   reward_mean:', reward_mean, '    epsilon: ', agent.epsilon
        break
    

if __name__ == '__main__':
  train_game()
  if EVAL_FLAG:
    gym.upload('/tmp/' + MODE_NAME, api_key='sk_nXYWtyR0CfjmTgSiJVJA')
 