import argparse
import gym
import time
import logger
import os, sys

import numpy as np
import tensorflow as tf
import utils as U
import monitor as M

from sac import SAC 
from collections import deque

parser = argparse.ArgumentParser(description='prioritized deep deterministic policy gradient algorithm')

parser.add_argument(
    '--train', action='store_true')

parser.add_argument(
    '--lr', default=1e-3, type=float, help='learning rate')

parser.add_argument(
    '--gamma', default=.99, type=float, help='gamma')

parser.add_argument(
    '--nenvs', default=12, type=int, help='the number of processes')

parser.add_argument(
    '--batch_size', default=100, type=int, help='training batch size')

parser.add_argument(
    '--replay_size', default=1000000, type=int, help='the size of replay buffer')

parser.add_argument(
    '--save', default=False, type=bool, help='whether to save network')

parser.add_argument(
    '--load', default=False, type=bool, help='whether to load network')

parser.add_argument(
    '--gym_id', default='HalfCheetah-v3', type=str, help='gym id')

parser.add_argument(
    '--seed', default=0, type=int)

args = parser.parse_args()

class PlayGym(object):
    def __init__(self, 
                 args, 
                 train_env, test_env, 
                 agent):

        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent

        self.nsteps = 200

        nenv = train_env.num_envs
        self.nenvs = nenv
        self.obs = np.zeros((nenv,) + train_env.observation_space.shape)
        self.obs[:] = train_env.reset()
        self.dones = [False for _ in range(nenv)]

    def learn(self):
        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        for update in range(1, 200000):
            tstart = time.time()

            outs, epinfos = self._run()
            epinfobuf.extend(epinfos)

            tnow = time.time()
            fps = int(5 * self.nsteps*self.nenvs / (tnow - tstart))

            logger.logkv("serial_timesteps", 5 * update*self.nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", 5 * update*self.nsteps*self.nenvs)
            logger.logkv("fps", fps)
            logger.logkv("loss_pi", outs[0])
            logger.logkv("loss_q1", outs[1])
            logger.logkv("loss_q2", outs[2])
            logger.logkv("loss_v", outs[3])       
            logger.logkv('eprewmean', U.safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', U.safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            logger.dumpkvs()

            if update % 100 == 0 and logger.get_dir():
                checkdir = os.path.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = os.path.join(checkdir, '%.5i'%update)
                print('Saving to', savepath)
                self.agent.save_net(savepath)

            if update % 10 == 0:
                self.play()

    def _run(self):
        epinfos = []
        for _ in range(5):
            for _ in range(self.nsteps):
                acts = self.agent.action(self.obs)
                new_obs, rews, self.dones, infos = self.train_env.step(acts)
                self.agent.perceive(self.obs, acts, rews, new_obs, self.dones)
                self.obs[:] = new_obs

                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)

            for _ in range(1000):
                outs = self.agent.train()

        return outs, epinfos

    def play(self):
        obs = self.test_env.reset()
        done = False
        total_rew = 0
        while not done:
            act = self.agent.action(obs)[0]
            # self.test_env.render()
            obs, rew, done, _ = self.test_env.step(act)
            total_rew += rew
        
        print("reward: {}".format(total_rew))

class MakeEnv(object):
    def __init__(self, curr_path):
        self.curr_path = curr_path

    def make(self, train=True):
        if train:
            return self.make_train_env()
        else:
            return self.make_test_env()

    def make_train_env(self):
        logger.configure(dir='{}/log'.format(curr_path))
        def make_env(rank):
            def _thunk():
                env = gym.make(args.gym_id)
                env.seed(args.seed + rank)
                env = M.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                return env
            return _thunk

        nenvs = args.nenvs
        env = U.SubprocVecEnv([make_env(i) for i in range(nenvs)])
        env = U.VecNormalize(env, ob=False, ret=False)

        return env

    def make_test_env(self):
        def make_env():
            env = gym.make(args.gym_id)
            return env

        env = U.DummyVecTestEnv([make_env])
        env = U.VecNormalizeTest(env)
        return env


if __name__ == '__main__':
    curr_path = sys.path[0]
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    maker = MakeEnv(curr_path) 
    train_env = maker.make_train_env()
    test_env = maker.make_test_env()

    ob_space = train_env.observation_space
    ac_space = train_env.action_space
    print(ac_space.high[0])

    agent = SAC(session, args, 
        ob_space.shape[0], ac_space.shape[0])

    player = PlayGym(
        args, train_env, test_env, agent
        )

    session.run(tf.global_variables_initializer())
    session.run(agent.target_init)
    if args.train:
        player.learn()
    else:
        player.play()
