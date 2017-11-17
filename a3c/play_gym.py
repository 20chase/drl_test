#! /usr/bin/env python3
import argparse
import gym
import roboschool
import scipy.signal

import numpy as np
import tensorflow as tf

from discrete_a2c import DiscreteA2C

parser = argparse.ArgumentParser(description='discrete advantage actor critic algorithm')

parser.add_argument(
    '--actor_lr', default=1e-4, type=float, help='actor learning rate')

parser.add_argument(
    '--critic_lr', default=1e-3, type=float, help='critic learning rate')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--lamb', default=0.95, type=float, help='GAE hyper parameter')

parser.add_argument(
    '--train_epochs', default=10, type=float, help='the training epochs')

parser.add_argument(
    '--batch_size', default=64, type=int, help='the number of trajectories')

parser.add_argument(
    '--max_steps', default=100000, type=int, help='max steps of training')

parser.add_argument(
    '--animate', default=False, type=bool, help='whether to animate environment')

parser.add_argument(
    '--huber', default=True, type=bool, help='whether to use huber loss')

parser.add_argument(
    '--save_network', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load_network', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--test_alg', default=False, type=bool, help='whether to test our algorithm')

parser.add_argument(
    '--gym_id', default='CartPole-v1', type=str, help='gym id')

parser.add_argument(
    '--model_name', default='discrete_a2c', type=str, help='save or load model name')

args = parser.parse_args()


class PlayGym(object):
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent

    def play(self, max_iters=1000):
        for i in range(max_iters):
            trajs = self._run_policy()
            obses, acts, advs, rets, rews = self._process_trajs(trajs)
            self.agent.update(obses, acts, advs, rets)
            stats = self.agent.show(obses, acts, advs, rets, rews)
            self._print_stats(stats)

    def _run_policy(self):
        trajs = []
        for e in range(self.args.batch_size):
            obses, acts, rews = self._run_episode()
            traj = {
            'obses': obses,
            'acts': acts,
            'rews': rews 
            }
            trajs.append(traj)

        return trajs

    def _run_episode(self):
        state = self.env.reset()
        obses, acts, rews = [], [], []
        done = False
        while not done:
            if self.args.animate:
                self.env.render()
            obses.append(state)
            action = self.agent.action(state)
            state, reward, done, info = self.env.step(self._convert_act(action))
            acts.append(action)
            rews.append(reward)

        return np.asarray(obses), np.asarray(acts), np.asarray(rews)

    def _convert_act(self, act):
        return np.argmax(act)

    def _process_trajs(self, trajs):
        trajs = self._add_value(trajs)
        trajs = self._add_ret(trajs)
        trajs = self._add_gae(trajs)
        return self._build_train_set()

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1] 

    def _add_value(self, trajs):
        for traj in trajs:
            obses = traj['obses']
            values = self.agent.get_value(obses)
            traj['values'] = np.squeeze(np.asarray(values))

        return trajs

    def _add_ret(self, trajs):
        for traj in trajs:
            rews = traj['rews'] * (1 - self.args.gamma)
            rets = self.discount(rews, self.args.gamma)
            traj['rets'] = rets

        return trajs

    def _add_gae(self, trajs):
        for traj in trajs:
            rews = traj['rews'] * (1 - self.args.gamma)
            values = traj['values']
            # td error
            tds = rews - values + np.append(values[1:]*self.args.gamma, 0)
            advs = self.discount(tds, self.args.gamma*self.args.lamb)
            advs = np.asarray(advs)
            traj['advs'] = advs

        return trajs

    def _build_train_set(self, trajs):
        obses = np.concatenate([traj['obses'] for traj in self.trajs])
        acts = np.concatenate([traj['acts'] for traj in self.trajs])
        advs = np.concatenate([traj['advs'] for traj in self.trajs])
        rets = np.concatenate([traj['rets'] for traj in self.trajs])
        rews = np.concatenate([traj['rews'] for traj in self.trajs])
        # normalize advs
        advs = (advs-advs.mean()) / (advs.std()+1e-6)
        return obses, acts, advs, rets, rews

    def _print_stats(self, stats):
        print("*********** Iteration {} ************".format(stats["Iteration"]))
        table = []
        for k, v in stats.items():
            table.append([k, v])

        print(tabulate(table, tablefmt="grid"))

if __name__ == '__main__':
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    env = gym.make(args.gym_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DiscreteA2C(session, args, obs_dim, act_dim)

    player = PlayGym(args, env, agent)

    session.run(tf.global_variables_initializer())

    player.play()


