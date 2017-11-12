#! /usr/bin/env python3
import ray
import gym

import numpy as np

@ray.remote
class RayEnvironment(object):
    def __init__(self, env):
        self.env = env
        state = self.env.reset()
        self.shape = state.shape

    def step(self, action):
        if self.done:
            return [np.zeros(self.shape), 0.0, True]
        else:
            state, reward, done, info = self.env.step(action)
            self.done = done
            return [state, reward, done]

    def reset(self):
        self.done = False
        return self.env.reset()
        
    
def run_episode(envs):
    terminates = [False for _ in range(len(envs))]
    terminates_idxs = [0 for _ in range(len(envs))]

    states = [env.reset.remote() for env in envs]
    states = ray.get(states)
    while not all(terminates):
        next_step = [env.step.remote([0.]) for i, env in enumerate(envs)]
        next_step = ray.get(next_step)

        states = [batch[0] for batch in next_step]
        rewards = [batch[1] for batch in next_step]
        dones = [batch[2] for batch in next_step]

        for i, d in enumerate(dones):
            if d:
                terminates[i] = True
            else:
                terminates_idxs[i] += 1


if __name__ == '__main__':
    ray.init()
    env_name = 'Pendulum-v0'

    envs = [gym.make(env_name) for _ in range(8)]
    envs = [RayEnvironment.remote(envs[i]) for i in range(2)]

    for i in range(100000):
        trajectories = run_episode(envs)
        print (i)





