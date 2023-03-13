from utils.reward_functions import *
from stable_baselines3 import PPO
from PPO import Agent, make_env
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch.nn as nn
import supersuit as ss
from utils.functions import *
import os
import random
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'pz_environment.py')))


def dict_to_torch(d):
    obs = np.array([d[a] for a in d.keys()])
    return torch.reshape(torch.tensor(
        obs, dtype=torch.float32), (50, 5))


def action_to_dict(a, env):
    return dict(zip(env.agents, a))


def test_agent(num_episodes, agent, env, env_parameter_dict=False):

    # DEFINE PARAMETERS #### Only used if input not defined
    env_parameter_dict = env_parameter_dict if env_parameter_dict else {
        "num_particles": 50,
        "dimensions": 2,
        "iterations": 100,
        "reward_function": diff_reward_end_rew,
        "prod_mode": False,
    }

    #### MAIN LOOP ####
    tot_hist = []
    initials = []
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        action_size = []
        count = 0
        done = False

        while not done:
            obs = dict_to_torch(obs)
            action, probs, ent = agent.get_action(obs)
            action = action.squeeze()
            action = action_to_dict(action, env)
            obs, rewards, terminations, truncations, infos = env.step(action)
            done = True in terminations.values()
            action_size.append(action)
            count += 1

        tot_hist.append(infos['agent_1']['Best score'])
        initials.append(infos['agent_1']['Initial'])

    avg = sum(tot_hist)/len(tot_hist)
    avg_improvement = sum([100*tot_hist[i]/initials[i]
                          for i in range(num_episodes)])/len(range(num_episodes))
    print(f'Avg performance: {avg}, avg improvement: {avg_improvement}')
    plt.plot(range(len(tot_hist)), tot_hist)
    plt.show()
    return avg, avg_improvement
