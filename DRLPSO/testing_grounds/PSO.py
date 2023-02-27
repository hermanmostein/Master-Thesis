import os
import random
import torch

import wandb
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils.conversions import aec_to_parallel
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


def run_pso(num_episodes, env, env_parameter_dict):

    #### MAIN LOOP ####
    tot_hist = []
    initials = []
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        count = 0

        for step in range(env_parameter_dict["num_particles"]*env_parameter_dict["dimensions"]*env_parameter_dict["iterations"]):
            action = [[0.0]]

            env.step(action)
        tot_hist.append(env.f(env.global_best_pos))
        initials.append(env.best_initial)

    avg = sum(tot_hist)/len(tot_hist)
    avg_improvement = sum([100*tot_hist[i]/initials[i]
                          for i in range(num_episodes)])/len(range(num_episodes))
    print(f'Avg performance: {avg}, avg improvement: {avg_improvement}')
    return avg, avg_improvement
