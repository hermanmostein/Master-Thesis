from PPO import Agent
from pz_environment import Env
from testing_grounds.PSO import run_pso
from testing_grounds.test_agent import test_agent
from utils.reward_functions import *
import os

from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss

def main(agent, env_parameter_dict = None):

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
        "total_timesteps": 8000000
    }
    parameter_dict = {**env_parameter_dict, **hp_dict}


    agents = [f'agent_{num}' for num in range(
        1, env_parameter_dict["num_particles"]+1)]

    env = Env(
        agents=agents, reward_function=env_parameter_dict["reward_function"],
        dimensions=env_parameter_dict["dimensions"], iterations=env_parameter_dict["iterations"],
        prod_mode=parameter_dict["prod_mode"], use_agent=True)

    agent_env = aec_to_parallel(env)
    agent_env = ss.pettingzoo_env_to_vec_env_v1(agent_env)
    agent_env = ss.concat_vec_envs_v1(agent_env, 4)

    if hp_dict["prod_mode"]:
        wandb.init(project="DRLPSO", entity="hermanmostein", sync_tensorboard=False,
                   name="DRLPSO", config=parameter_dict, monitor_gym=True, save_code=True)
        os.environ["WANDB_SILENT"] = "true"

    #### TRAINING ####

    agent = Agent(env=agent_env, hp_dict=hp_dict)
    agent.train()
    if (hp_dict["prod_mode"]):
        agent.save_model()

    env = Env(
        agents=agents, reward_function=env_parameter_dict["reward_function"],
        dimensions=env_parameter_dict["dimensions"], iterations=env_parameter_dict["iterations"],
        prod_mode=parameter_dict["prod_mode"], use_agent=False)

    pso_obj, pso_imp = run_pso(50, env=env, env_parameter_dict = env_parameter_dict)

    env = Env(
        agents=agents, reward_function=env_parameter_dict["reward_function"],
        dimensions=env_parameter_dict["dimensions"], iterations=env_parameter_dict["iterations"],
        prod_mode=parameter_dict["prod_mode"], use_agent=True)
    drlpso_obj, drlpso_imp = test_agent(50, agent, env, env_parameter_dict = env_parameter_dict)


main(None)
    