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
            obs = torch.reshape(torch.tensor(
                obs, dtype=torch.float32), (1, 5))
            # print(f'Observation in step {count}: {obs}')
            action, probs, ent = agent.get_action(obs)
            # print(f'Action in step {count}: {action}')

            obs, rs, done, infos = env.step(torch.clamp(
                action, -1, 1))
            # print(obs)
            action_size.append(action)
            count += 1
            # print(f'Best found score: {env.f(env.global_best_pos)}')

            # obs = env.observations[env.agent_selection]
            # obs = env.observe(env.agent_selection)

        tot_hist.append(env.f(env.global_best_pos))
        initials.append(env.best_initial)

    plt.plot(range(len(tot_hist)), tot_hist)
    plt.show()
    avg = sum(tot_hist)/len(tot_hist)
    avg_improvement = sum([100*tot_hist[i]/initials[i]
                          for i in range(num_episodes)])/len(range(num_episodes))
    print(f'Avg performance: {avg}, avg improvement: {avg_improvement}')
    return avg, avg_improvement
