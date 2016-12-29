import tensorflow as tf
import tensorlayer as tl
import gym
import numpy as np


# hyber parameters
ACTIONS = 2
GAMMA = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 10
LOAD = False
SAVE = False
DISPLAY = False
MODE_NAME = 'CartPole-v0'

class Brain():
    def __init__(self, env):

        self.action = ACTIONS

        self.creat_network()

        self.train()

        self.session = tf.InteractiveSession()

        self.merged = tf.merge_all_summaries()

        self.train_writer = tf.train.SummaryWriter('/tmp/train',
                                              self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def creat_network(self):
        self.state_batch = tf.placeholder(tf.float32, shape=[None, 4])

        self.network = tl.layers.InputLayer(self.state_batch)
        self.network = tl.layers.DenseLayer(self.network, n_units=40, act=tf.nn.relu, name='relu0')
        self.network = tl.layers.DenseLayer(self.network, n_units=40, act=tf.nn.relu, name='relu1')
        self.network = tl.layers.DenseLayer(self.network, n_units=40, act=tf.nn.relu, name='relu2')
        self.network = tl.layers.DenseLayer(self.network, n_units=self.action)

        self.output = self.network.outputs

        self.output_prob = tf.nn.softmax(self.output)


    def train(self):
        self.action_batch = tf.placeholder(tf.int32, shape=[None])

        self.reward_batch = tf.placeholder(tf.float32, shape=[None])

        self.loss = tl.rein.cross_entropy_reward_loss(self.output, self.action_batch, self.reward_batch)

        self.reward_mean = tf.placeholder(tf.float32)

        with tf.name_scope('loss'):
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('reward'):
            tf.scalar_summary('reward_means', self.reward_mean)




        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)



def main():
    env = gym.make(MODE_NAME)
    agent = Brain(env)

    if LOAD is True:
        params = tl.files.load_npz(name=MODE_NAME + '.npz')
        tl.files.assign_params(agent.session, params, agent.network)

    state_batch, action_batch, reward_batch = [], [], []

    state = env.reset()


    state = state.reshape(1, 4)
    prob = agent.output_prob.eval(feed_dict={agent.state_batch: state})
    action = np.random.choice([0, 1], p=prob.flatten())

    episode = 0
    reward_sum = 0
    reward_mean = 0
    while True:
        if DISPLAY is True:
            env.render()

        next_state, reward, terminate, _ = env.step(action)
        state_batch.append(state)
        action_batch.append(action)

        reward_sum += reward


        state = next_state

        state = state.reshape(1, 4)
        prob = agent.output_prob.eval(feed_dict={agent.state_batch: state})
        action = np.random.choice([0, 1], p=prob.flatten())

        if terminate is True:

            reward_batch.append(reward_sum)

            episode += 1

            print 'episode: ', episode, '    reward_sum: ', reward_sum

            reward_mean += reward_sum

            if episode % BATCH_SIZE == 0:
                print ' ...... update parameters ...... '
                state_batch = np.vstack(state_batch)
                action_batch = np.asarray(action_batch)
                reward_batch = np.asarray(reward_batch)
                reward_batch = tl.rein.discount_episode_rewards(reward_batch, gamma=GAMMA)
                reward_batch -= np.mean(reward_batch)
                reward_batch /= np.std(reward_batch)

                reward_mean = reward_mean / BATCH_SIZE

                print 'reward_mean = ', reward_mean

                summary, _ = agent.session.run([agent.merged, agent.optimizer], feed_dict = {agent.state_batch: state_batch, agent.action_batch: action_batch,agent.reward_batch: reward_batch, agent.reward_mean: reward_mean})

                agent.train_writer.add_summary(summary, episode)


                if SAVE is True:
                    tl.files.save_npz(agent.network.all_params, name=MODE_NAME + '.npz')

                state_batch, action_batch, reward_batch = [], [], []
                reward_mean = 0

            state = env.reset()
            state = state.reshape(1, 4)
            prob = agent.output_prob.eval(feed_dict={agent.state_batch: state})
            action = np.random.choice([0, 1], p=prob.flatten())
            reward_sum = 0

        else:
            reward_batch.append(0)

if __name__ == '__main__':
    main()