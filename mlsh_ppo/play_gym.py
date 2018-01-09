#! /usr/bin/env python3
import argparse
import gym
import sys
import time
import os
import roboschool
import logger
import locomotionenv.envs

import monitor as M
import os.path as osp
import numpy as np
import tensorflow as tf
import utils as U

from collections import deque
from ppo_cliped import PPOCliped


parser = argparse.ArgumentParser(description='proximal policy optimization cliped version')

parser.add_argument(
    '--lr', default=3e-4, type=float, help='learning rate')

parser.add_argument(
    '--d_targ', default=0.0012, type=float, help='the target of kl divergence')

parser.add_argument(
    '--ent_coef', default=0., type=float, help='the coefficient of entropy')

parser.add_argument(
    '--clip_range', default=0.2, type=float, help='the clip range parameter')

parser.add_argument(
    '--vf_coef', default=0.5, type=float, help='the coefficient of value function')

parser.add_argument(
    '--max_grad_norm', default=0.5, type=float, help='max gradients normalize')

parser.add_argument(
    '--lamb', default=.98, type=float, help='GAE hyper parameters')

parser.add_argument(
    '--gamma', default=.995, type=float, help='gamma')

parser.add_argument(
    '--seed', default=0, type=int, help='RNG seed')

parser.add_argument(
    '--num_batchs', default=4, type=int, help='the number of batchs')

parser.add_argument(
    '--num_opts', default=4, type=int, help='the number of opts')

parser.add_argument(
    '--save_interval', default=50, type=int, help='the number of save_network')

parser.add_argument(
    '--num_steps', default=512, type=int, help='the number of steps')

parser.add_argument(
    '--num_procs', default=32, type=int, help='the number of processes')

parser.add_argument(
    '--max_steps', default=10e6, type=int, help='max steps of training')

parser.add_argument(
    '--train', default=True, type=bool, help='Whether to train')

parser.add_argument(
    '--point', default='00500', type=str, help='the point for loading')

parser.add_argument(
    '--gym_id', default='Ant-v1', type=str, help='gym id')

args = parser.parse_args()

class PlayGym(object):
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent
        self.total_timesteps = self.args.max_steps
        self.nminibatches = self.args.num_batchs
        self.nsteps = self.args.num_steps
        self.gamma = self.args.gamma
        self.lam = self.args.lamb
        self.noptepochs = self.args.num_opts
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(nenv)]

    def learn(self):
        env = self.env
        nsteps = self.nsteps
        nminibatches = self.nminibatches
        total_timesteps = self.total_timesteps
        total_timesteps = int(total_timesteps)
        noptepochs = self.noptepochs
        save_interval = self.args.save_interval

        loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        nenvs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        print (" ------------- Shape --------------- ")
        print ("obs_dim: {} | ac_dim: {}".format(ob_space.shape[0], ac_space.shape[0]))
        print (" ----------------------------------- ")
        nbatch = nenvs * nsteps
        nbatch_train = nbatch // nminibatches

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        lrnow = self.args.lr
        cliprangenow = self.args.clip_range
        nupdates = total_timesteps//nbatch
        init_targ = self.args.d_targ
        kl = 0.01

        def adaptive_lr(lr, kl, d_targ):
            if kl < (d_targ / 1.5):
                lr *= 2.
            elif kl > (d_targ * 1.5):
                lr *= .5
            return lr

        for update in range(1, nupdates+1):
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            curr_step = update*nbatch
            step_percent = float(curr_step / total_timesteps)

            if step_percent < 0.1:
                d_targ = init_targ
            elif step_percent < 0.4:
                d_targ = init_targ / 2.
            else:
                d_targ = init_targ / 4.

            lrnow = adaptive_lr(lrnow, kl, d_targ)

            obs, returns, masks, actions, values, neglogpacs, states, epinfos = self._run()
            epinfobuf.extend(epinfos)
            mblossvals = []

            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs))
                    mblossvals.append(self.agent.train(lrnow, cliprangenow, *slices))
                       
            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            kl = lossvals[3]
            if update % 1 == 0 or update == 1:
                ev = U.explained_variance(values, returns)
                logger.logkv("serial_timesteps", update*nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', U.safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', U.safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                logger.logkv('lr', lrnow)
                logger.logkv('d_targ', d_targ)
                for (lossval, lossname) in zip(lossvals, loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()

            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%update)
                print('Saving to', savepath)
                self.agent.save_network(savepath)
                np.save('{}/mean'.format(logger.get_dir()), self.env.ob_rms.mean)
                np.save('{}/var'.format(logger.get_dir()), self.env.ob_rms.var)
        
        env.close()

    def _run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = None
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, neglogpacs = self.agent.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)            
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.agent.get_value(self.obs)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0        
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(U.sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), 
            mb_states, epinfos)

    def play(self, curr_path):
        self.agent.load_network("{}/log/checkpoints/{}".format(curr_path, self.args.point))
        def run_episode():
            obs = self.env.reset()
            score = 0
            done = [False]
            while not done[0]:
                env.render()
                act = self.agent.get_action(obs)
                obs, rew, done, info = env.step(act)
                score += rew[0]

            return score

        for e in range(10000):
            score = run_episode()
            print ('episode: {} | score: {}'.format(e, score))

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

        nenvs = args.num_procs
        env = U.SubprocVecEnv([make_env(i) for i in range(nenvs)])
        env = U.VecNormalize(env)

        return env

    def make_test_env(self):
        def make_env():
            env = gym.make(args.gym_id)
            return env

        env = U.DummyVecTestEnv([make_env])
        running_mean = np.load('{}/log/mean.npy'.format(self.curr_path))
        running_var = np.load('{}/log/var.npy'.format(self.curr_path))
        env = U.VecNormalizeTest(env, running_mean, running_var)
        return env

if __name__ == '__main__':
    curr_path = sys.path[0]
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    # make env
    maker = MakeEnv(curr_path) 
    env = maker.make(args.train)
    
    # build agent
    ob_space = env.observation_space
    ac_space = env.action_space
    agent = PPOCliped(session, args, ob_space, ac_space)

    # build player
    player = PlayGym(args, env, agent)

    # start
    session.run(tf.global_variables_initializer())
    if args.train:
        player.learn()
    else:
        player.play(curr_path)