import os
import random
import torch

import wandb
from pz_environment import Env, new_env
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils.conversions import aec_to_parallel
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
from utils.functions import *


def main(num_episodes, use_agent=True):

    #### DEFINE PARAMETERS ####
    env_parameter_dict = {
        "num_particles": 50,
        "dimensions": 2,
        "iterations": 100,
        "reward_function": final_score
    }
    hp_dict = {
        "prod_mode": True,
        "gamma": 1,
        "learning_rate": 5e-4,
        "clip_coef": 0.2,
        "total_timesteps": 20000000
    }
    parameter_dict = {**env_parameter_dict, **hp_dict}

    #### PREPARE ENVIRONMENT ####
    num_envs = 4
    agents = [f'agent_{num}' for num in range(
        1, env_parameter_dict["num_particles"]+1)]
    env = new_env(
        agents=agents, reward_function=env_parameter_dict["reward_function"],
        dimensions=env_parameter_dict["dimensions"], iterations=env_parameter_dict["iterations"],
        prod_mode=parameter_dict["prod_mode"])

    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4)

    #### LOGGING ####
    if hp_dict["prod_mode"]:
        wandb.init(project="DRLPSO", entity="hermanmostein", sync_tensorboard=False,
                   name="DRLPSO", config=parameter_dict, monitor_gym=True, save_code=True)
        os.environ["WANDB_SILENT"] = "true"

    #### TRAINING ####
    if (use_agent):

        agent = Agent(env=env, hp_dict=hp_dict)
        agent.train()
        if (hp_dict["prod_mode"]):
            agent.save_model()

        for e in env.vec_envs:
            e.__setattr__('set_mode', 'Normal')

    #### MAIN LOOP ####
    tot_hist = []

    env = new_env(
        agents=agents, reward_function=env_parameter_dict["reward_function"], dimensions=env_parameter_dict[
            "dimensions"], iterations=env_parameter_dict["iterations"],
        prod_mode=False, use_agent=use_agent)

    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1)

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = [False for n in range(num_envs)]
        count = 0

        for step in range(parameter_dict["num_particles"]*parameter_dict["dimensions"]*parameter_dict["iterations"]):
            if (use_agent):
                with torch.no_grad():
                    obs = torch.from_numpy(obs).float()
                    action, probs, ent = agent.get_action(obs)
            else:
                action = [[0.0]
                          for a in range(env_parameter_dict["num_particles"])]

            obs, reward, done, _, info = env.step(action)
    print(f'Last global best: {env.get_attr("global_best_pos")[0]}')

    avg = sum(tot_hist)/len(tot_hist)
    print(f'Avg performance: {avg}')
    env.close()
    return avg, griewangk(np.array([0, 0]))


a_avg, global_min = main(20, use_agent=True)
no_a_avg, global_min = main(20, use_agent=False)

print(
    f'No agent: {no_a_avg}, With agent: {a_avg}, Global minimum: {global_min}')
