from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Box
import functools

from array import array
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import gym
from gym import spaces
import numpy as np
import torch
from utils.functions import *
from pettingzoo.utils.conversions import parallel_wrapper_fn
import wandb


def new_env(**kwargs):
    env = Env(**kwargs)
    return env


class Env(AECEnv):

    metadata = {"render_modes": ["human"],
                "name": "pso", "is_parallelizable": True}

    def __init__(self, agents, reward_function, func=griewangk, use_agent=True, prod_mode=False, dimensions=2, iterations=100, omega=0.9, phi_p=2, phi_g=2):

        self.num_particles = len(agents)
        self.dimensions = dimensions
        self.max_iterations = iterations
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.f = func
        self.use_agent = use_agent
        self.prod_mode = prod_mode
        self.reward_function = reward_function

        self.restrictions = self.f(None)
        self.particle_pos = np.array([[self.restrictions[i][1]*(2*random.random()-1) for i in range(dimensions)]
                                      for j in range(self.num_particles)])
        self.particle_prev_pos = copy.deepcopy(self.particle_pos)
        self.particle_pos_hist = [[] for i in range(self.num_particles)]
        self.particle_velocity = [[random.random() for i in range(
            dimensions)] for j in range(self.num_particles)]
        self.particle_best_pos = copy.deepcopy(self.particle_pos)
        self.particle_best_score = np.array(
            [self.f(c) for c in self.particle_pos])

        min_particle = np.argmin(
            [self.f(self.particle_best_pos[particle]) for particle in range(self.num_particles)])
        self.global_best_pos = copy.deepcopy(
            self.particle_best_pos[min_particle])
        self.global_best_hist = [copy.deepcopy(self.global_best_pos)]
        self.particle_scores = np.array([self.f(c) for c in self.particle_pos])
        self.iteration = 0
        self.reward = 0
        self.particle = 0
        self.dim = 0
        self.mode = 'Normal'
        self.function_list = [translated_rastrigin]  # [rosenbrock, rastrigrin,
        # zakharov, schwefel, ackley, translated_rastrigin, translated_ackley]
        self.track_training = []
        self.initial_scores = copy.deepcopy(self.particle_scores)
        self.improvement = 0
        self.performances = {}
        self.s = random.random()

        self.possible_agents = agents
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self._agent_selector.reset()

        self.action_spaces = {agent: Box(
            low=-1, high=1, shape=(1,)) for agent in self.possible_agents}
        self.observation_spaces = {agent: Box(
            low=-1, high=1, shape=(5,)) for agent in self.possible_agents}
        self.render_mode = None

    '''@functools.lru_cache(maxsize=None)
    def observation_space(self, agent=0):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(
            low=-1, high=2, shape=(5,))

    @ functools.lru_cache(maxsize=None)
    def action_space(self, agent=0):
        return Box(
            low=-1, high=1, shape=(1,))
'''

    def reset(self, seed=None, return_info=False, options=None):

        if (self.mode == 'Train'):

            self.s = random.random()
            get_func = random.sample(self.function_list, 1)[0]
            self.f = get_func(self.s)
        else:
            self.f = rastrigrin

        self.restrictions = self.f(None)

        self.particle_pos = np.array([[min(self.restrictions[i][1], self.restrictions[i][0])*(2*random.random()-1) for i in range(self.dimensions)]
                                      for j in range(self.num_particles)])
        self.particle_velocity = [[random.random() for i in range(
            self.dimensions)] for j in range(self.num_particles)]
        self.particle_pos_hist = [[] for i in range(self.num_particles)]
        self.particle_best_pos = copy.deepcopy(self.particle_pos)
        self.particle_best_score = np.array(
            [self.f(c) for c in self.particle_pos])
        self.particle_prev_pos = copy.deepcopy(self.particle_pos)
        min_particle = np.argmin(list([self.f(
            self.particle_best_pos[particle]) for particle in range(self.num_particles)]))
        self.global_best_pos = copy.deepcopy(
            self.particle_best_pos[min_particle])
        self.global_best_hist = [copy.deepcopy(self.global_best_pos)]
        self.particle_scores = np.array([self.f(c) for c in self.particle_pos])
        self.iteration = 0
        self.reward = {a: 0 for a in self.possible_agents}
        self.dim = 0
        self.initial_scores = copy.deepcopy(self.particle_scores)
        self.best_initial = min(self.initial_scores)
        self.initial_pos = copy.deepcopy(self.particle_pos)
        self.mean_initial = np.mean(self.initial_scores)
        observation = self.make_observation(0, 0)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: [0] for agent in self.agents}
        self.observations = {agent: self.make_observation(
            self.agents.index(agent), 0) for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.T_0 = self.mean_initial
        self.T = self.T_0
        self.alpha = 0.99
        self.done = False

        return observation

    def observe(self, agent):

        return np.array(self.observations[agent])

    def step(self, action):

        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            return self._was_dead_step(action)
        self.particle_pos_hist[self.particle].append(
            copy.deepcopy(self.particle_pos[self.particle]))

        _dim = self.dim
        # self.agent_selection - 1
        particle = int(self.agent_selection[-1]) - 1
        agent = self.agent_selection
        self.state[agent] = action
        velocity_cap = np.sqrt(2)*abs(self.restrictions[0]
                                      [0] - self.restrictions[0][1])/40
        '''r = self.iteration/self.max_iterations
        velocity_cap *= 1 / (1 + np.exp(10*r - 5))'''

        if (action is not None and self.use_agent):

            # This is so the agent can make smaller steps
            velocity = np.sign(action) * action**2
            velocity *= velocity_cap

        # Handles normal pso
        else:

            r_g = random.random()
            r_p = random.random()

            velocity = self.omega*self.particle_velocity[particle][_dim] + self.phi_p*r_p*(copy.deepcopy(self.particle_best_pos[particle][_dim]) - copy.deepcopy(
                self.particle_pos[particle][_dim])) + self.phi_g*r_g*(copy.deepcopy(self.global_best_pos[_dim]) - copy.deepcopy(self.particle_pos[particle][_dim]))
            velocity = max(min(velocity, velocity_cap), -velocity_cap)

        # Update positions
        self.particle_prev_pos[particle][_dim] = copy.deepcopy(
            self.particle_pos[particle][_dim])
        self.particle_velocity[particle][_dim] = velocity
        self.particle_pos[particle][_dim] += self.particle_velocity[particle][_dim]

        self.particle_pos[particle][_dim] = min(
            copy.deepcopy(self.particle_pos[particle][_dim]), self.restrictions[_dim][1])
        self.particle_pos[particle][_dim] = max(
            copy.deepcopy(self.particle_pos[particle][_dim]), self.restrictions[_dim][0])

        self.get_reward(particle, _dim)

        if (self._agent_selector.is_last()):
            self.dim += 1
        if (self.dim >= self.dimensions):
            self.dim = 0

        self.rewards[agent] = self.reward[agent]

        observation = self.make_observation(particle=particle, dim=_dim)
        self.observations[agent] = observation
        done = self.iteration > self.max_iterations
        self.done = done
        info = {'global_best_score': self.f(
            self.global_best_pos), 'best_initial': self.best_initial}

        if (done):
            if (self._agent_selector.is_last()):
                self.terminations = {agent: done for agent in self.agents}
            self.track_training.append(self.f(self.global_best_pos))
            score = self.f(self.global_best_pos)
            self.improvement = 100*score/self.best_initial
            if (self.prod_mode):
                wandb.log({"status/improvement": self.improvement})
                wandb.log({"status/objective": score})

        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()
        # if (dim >= self.dimensions-1):
        self.agent_selection = self._agent_selector.next()

        if (self._agent_selector.is_last() and _dim == self.dimensions-1):
            self.iteration += 1
            self.T *= self.alpha

        return self.observations[self.agent_selection], self.rewards[self.agent_selection], self.done, info

    def close(self):
        return None

    def get_reward(self, particle, dim):

        agent = self.agent_selection
        self.reward[agent] = 0
        if self.dim < self.dimensions-1:
            self._clear_rewards()
            return

        score_ = self.f(self.particle_pos[particle])
        self.particle_scores[particle] = score_

        self.reward_function(self, particle)

        if (score_ < self.f(self.particle_prev_pos[particle])):

            if (score_ < self.f(self.particle_best_pos[particle])):
                self.particle_best_pos[particle] = copy.deepcopy(
                    self.particle_pos[particle])
                self.particle_best_score[particle] = self.f(
                    self.particle_best_pos[particle])

                if (score_ < self.f(self.global_best_pos)):
                    self.global_best_pos = copy.deepcopy(
                        self.particle_pos[particle])
                    self.global_best_hist.append(
                        copy.deepcopy(self.particle_pos[particle]))

    def make_observation(self, particle, dim):

        res = abs(self.restrictions[dim][0] - self.restrictions[dim][1])
        initial_score = self.mean_initial
        ratio = self.f(self.particle_pos[particle])/initial_score

        pos = self.particle_pos[particle][dim]/res
        global_best_pos = self.global_best_pos[dim]/res
        best_pos = self.particle_best_pos[particle][dim]/res
        iter = self.iteration/self.max_iterations
        v = self.particle_velocity[particle][dim]/res

        state = [best_pos-pos, global_best_pos-pos,
                 ratio, iter, v]

        return np.clip(np.array(state, dtype=np.float32).flatten(), -1, 2)

    def set_mode(self, mode):
        self.mode = mode

    def plot_training(self):
        N = 2
        m_avg = np.convolve(self.track_training, np.ones(N)/N, mode='valid')
        plt.plot(range(len(self.track_training)),
                 self.track_training, color='r', label='Best objective')
        plt.plot(range(len(m_avg)), m_avg,
                 color='blue', label='Moving average')
        plt.legend()
        plt.title('Best objective for episode during training')
        plt.show()

    def render():
        pass
