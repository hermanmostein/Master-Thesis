import numpy as np
import wandb


def one_three_ten(env, particle):

    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    env.particle_scores[particle] = score_
    diff = env.f(env.particle_prev_pos[particle]) - score_

    ratio = score_/env.mean_initial

    if (diff > 0):
        env.reward[agent] += 1
        if (score_ < env.f(env.particle_best_pos[particle])):

            env.reward[agent] += 2

            if (score_ < env.f(env.global_best_pos)):
                env.reward[agent] += 7


def one_three_log(env, particle):

    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    diff = env.f(env.particle_prev_pos[particle]) - score_

    ratio = score_/env.mean_initial

    if (diff > 0):
        if (score_ < env.f(env.particle_best_pos[particle])):

            env.reward[agent] += 1

            if (score_ < env.f(env.global_best_pos)):
                env.reward[agent] += 2 - 100*np.log(ratio)


def one_three_ten_bonus(env, particle):

    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    env.particle_scores[particle] = score_
    old = env.f(env.particle_prev_pos[particle])
    diff = old - score_
    wandb.log({"Improvement in this particlestep": diff})

    ratio = score_/env.mean_initial

    if (diff > 0):
        env.reward[agent] += 1
        if (score_ < env.f(env.particle_best_pos[particle])):

            env.reward[agent] += 2

            if (score_ < env.f(env.global_best_pos)):
                env.reward[agent] += 7 + (diff/old)*15


def one_three_five_bonus(env, particle):

    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    env.particle_scores[particle] = score_
    old = env.f(env.particle_prev_pos[particle])
    diff = old - score_

    ratio = score_/env.mean_initial

    if (diff > 0):
        env.reward[agent] += 1
        if (score_ < env.f(env.particle_best_pos[particle])):

            env.reward[agent] += 2

            if (score_ < env.f(env.global_best_pos)):
                env.reward[agent] += 2 + (diff/old)*15


def diff_reward(env, particle):

    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    env.particle_scores[particle] = score_
    old = env.f(env.particle_prev_pos[particle])
    diff = old - score_

    ratio = score_/env.mean_initial

    if (diff > 0):
        env.reward[agent] += 1
        if (score_ < env.f(env.particle_best_pos[particle])):

            env.reward[agent] += 2 + (diff/old)*10

            if (score_ < env.f(env.global_best_pos)):
                env.reward[agent] += 2 + (diff/old)*15


def diff_reward_end_rew(env, particle):

    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    env.particle_scores[particle] = score_
    old = env.f(env.particle_prev_pos[particle])
    diff = old - score_

    ratio = score_/env.mean_initial

    if (diff > 0):
        env.reward[agent] += 1
        if (score_ < env.f(env.particle_best_pos[particle])):

            env.reward[agent] += 2 + (diff/old)*5

            if (score_ < env.f(env.global_best_pos)):
                env.reward[agent] += 2 + (diff/old)*5

    if (env.iteration == env.max_iterations):
        env.reward[agent] += 10/(ratio+5e-4)


def final_score(env, particle):
    agent = env.agent_selection
    score_ = env.f(env.particle_pos[particle])
    env.particle_scores[particle] = score_

    ratio = score_/env.mean_initial

    if (env.iteration == env.max_iterations):
        env.reward[agent] += 1/(ratio+5e-4)
