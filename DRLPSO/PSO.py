import os
import random
import torch

import wandb
from environment import PSO
from pz_environment import Env, new_env
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils.conversions import aec_to_parallel
from functions import *
from callback import TensorboardCallback
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


def run_pso(num_episodes, env_parameter_dict=False):

    #### DEFINE PARAMETERS ####
    env_parameter_dict = env_parameter_dict if env_parameter_dict else {
        "num_particles": 50,
        "dimensions": 2,
        "iterations": 100,
        "reward_function": one_three_ten_bonus,
        "prod_mode": False,
    }

    #### PREPARE ENVIRONMENT ####
    agents = [f'agent_{num}' for num in range(
        1, env_parameter_dict["num_particles"]+1)]
    env = Env(
        agents=agents, reward_function=env_parameter_dict["reward_function"],
        dimensions=env_parameter_dict["dimensions"], iterations=env_parameter_dict["iterations"],
        prod_mode=env_parameter_dict["prod_mode"], use_agent=False)

    #### MAIN LOOP ####
    tot_hist = []
    initials = []
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        count = 0

        for step in range(env_parameter_dict["num_particles"]*env_parameter_dict["dimensions"]*env_parameter_dict["iterations"]):
            action = [[0.0]
                      for a in range(env_parameter_dict["num_particles"])]

            env.step(action)
        tot_hist.append(env.f(env.global_best_pos))
        initials.append(env.best_initial)

    avg = sum(tot_hist)/len(tot_hist)
    avg_improvement = sum([100*tot_hist[i]/initials[i]
                          for i in range(num_episodes)])
    print(f'Avg performance: {avg}')
    return avg, griewangk(np.array([0, 0]))


a_avg, global_min = main(20, use_agent=False)
