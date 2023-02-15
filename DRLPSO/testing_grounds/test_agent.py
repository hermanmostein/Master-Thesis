import os
import random
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'pz_environment.py')))
from utils.functions import *
import supersuit as ss
import torch.nn as nn
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PPO import Agent, make_env
from stable_baselines3 import PPO
import supersuit as ss
from utils.reward_functions import *


def test_agent(num_episodes, agent, env, env_parameter_dict=False):

    #### DEFINE PARAMETERS ####
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
        done = False
        count = 0

        while not env.done:

            obs = env.observe(env.agent_selection)
            obs = torch.reshape(torch.tensor(obs), (1,5))
            action, probs, ent = agent.get_action(obs)

            env.step(action)
        tot_hist.append(env.f(env.global_best_pos))
        initials.append(env.best_initial)

    avg = sum(tot_hist)/len(tot_hist)
    avg_improvement = sum([100*tot_hist[i]/initials[i]
                          for i in range(num_episodes)])/len(range(num_episodes))
    print(f'Avg performance: {avg}, avg improvement: {avg_improvement}')
    return avg, avg_improvement


#100 eps 10 dims, 23.834576