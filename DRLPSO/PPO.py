import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
from gymnasium.spaces import Box
import time
import random
from stable_baselines3.common.vec_env import VecEnvWrapper
import wandb

default_hp_dict = {
    "prod_mode": False,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "clip_coef": 0.1,
    "total_timesteps": 2000000
}

s = str(int(time.time()))


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        temp = self.venv.step_wait()
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        temp = self.venv.step_wait()
        obs, rews, dones, _, infos = temp
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(
                    time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


def make_env(env, device):
    # deal with dm_control's Dict observation space
    env = VecMonitor(env)
    env = VecPyTorch(env, device)

    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env, hp_dict=default_hp_dict, channels=3):
        super(Agent, self).__init__()
        self.parser = argparse.ArgumentParser(description='PPO agent')
    # Coself.mmon arguments
        self.parser.add_argument('--exp-name', type=str, default=f'DRLPSO__{s}',
                                 help='the name of this experiment')
        self.parser.add_argument('--gym-id', type=str, default="PPO",
                                 help='the id of the gym environment')
        self.parser.add_argument('--learning-rate', type=float, default=hp_dict["learning_rate"],
                                 help='the learning rate of the optimizer')
        self.parser.add_argument('--seed', type=int, default=1,
                                 help='seed of the experiment')
        self.parser.add_argument('--total-timesteps', type=int, default=hp_dict["total_timesteps"],
                                 help='total timesteps of the experiments')
        self.parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help='if toggled, `torch.backends.cudnn.deterministic=False`')
        self.parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help='if toggled, cuda will not be enabled by default')

        self.parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=hp_dict["prod_mode"], nargs='?', const=True,
                                 help='run the script in production mode and use wandb to log outputs')

        self.parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                                 help='weather to capture videos of the agent performances (check out `videos` folder)')
        self.parser.add_argument('--wandb-project-name', type=str, default="DRLPSO",
                                 help="the wandb's project name")
        self.parser.add_argument('--wandb-entity', type=str, default="hermanmostein",
                                 help="the entity (team) of wandb's project")
        # Algorithm specific arguments
        self.parser.add_argument('--n-minibatch', type=int, default=32,
                                 help='the number of mini batch')
        self.parser.add_argument('--num-envs', type=int, default=4,
                                 help='the number of parallel game environment')
        self.parser.add_argument('--num-steps', type=int, default=128,
                                 help='the number of steps per game environment')
        self.parser.add_argument('--gamma', type=float, default=hp_dict["gamma"],
                                 help='the discount factor gamma')
        self.parser.add_argument('--gae-lambda', type=float, default=0.95,
                                 help='the lambda for the general advantage estimation')
        self.parser.add_argument('--ent-coef', type=float, default=0.01,
                                 help="coefficient of the entropy")
        self.parser.add_argument('--vf-coef', type=float, default=0.5,
                                 help="coefficient of the value function")
        self.parser.add_argument('--max-grad-norm', type=float, default=0.5,
                                 help='the maximum norm for the gradient clipping')
        self.parser.add_argument('--clip-coef', type=float, default=hp_dict["clip_coef"],
                                 help="the surrogate clipping coefficient")
        self.parser.add_argument('--update-epochs', type=int, default=10,
                                 help="the K epochs to update the policy")
        self.parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help='If toggled, the policy updates will be early stopped w.r.t target-kl')
        self.parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                                 help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
        self.parser.add_argument('--target-kl', type=float, default=0.03,
                                 help='the target-kl variable that is referred by --kl')
        self.parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help='Use GAE for advantage computation')
        self.parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help="Toggles advantages normalization")
        self.parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help="Toggle learning rate annealing for policy and value networks")
        self.parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                                 help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

        self.args = self.parser.parse_args()
        if not self.args.seed:
            self.args.seed = int(time.time())

        self.hp_dict = hp_dict

        experiment_name = f"{self.args.gym_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{experiment_name}")
        self.writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(self.args).items()])))

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        # petting zoo

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   and self.args.cuda else 'cpu')
        self.env = make_env(env=env, device=self.device)

        self.args.num_envs = self.env.num_envs
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(
            self.args.batch_size // self.args.n_minibatch)

        self.network = nn.Sequential(
            layer_init(nn.Linear(5, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),

        )
        self.actor_mean = layer_init(
            nn.Linear(64, np.prod(self.env.action_space.shape)), std=1e-5)
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(self.env.action_space.shape)))
        self.critic = layer_init(nn.Linear(64, 1), std=1)

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.args.learning_rate, eps=1e-5)

        if (self.args.prod_mode):
            wandb.config.update(self.args)

        # ALGO Logic: Storage for epoch data
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) +
                               self.env.observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) +
                                   self.env.action_space.shape).to(self.device)
        self.logprobs = torch.zeros(
            (self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros(
            (self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros(
            (self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros(
            (self.args.num_steps, self.args.num_envs)).to(self.device)
        assert isinstance(self.env.action_space,
                          Box), "only continuous action space is supported"

    def lr(self, f): return f * self.args.learning_rate

    def get_action(self, x, action=None):

        action_mean = self.actor_mean(self.network(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def train(self):
        global_step = 0
        start_time = time.time()
        # Note how `next_obs` and `next_done` are used; their usage is equivalent to
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
        next_obs = self.env.reset()
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size
        for update in range(1, num_updates+1):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = self.lr(frac)
                self.optimizer.param_groups[0]['lr'] = lrnow

            # TRY NOT TO MODIFY: prepare the execution of the game.
            for step in range(0, self.args.num_steps):
                global_step += 1 * self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    self.values[step] = self.get_value(
                        self.obs[step]).flatten()
                    action, logproba, _ = self.get_action(self.obs[step])

                self.actions[step] = action
                self.logprobs[step] = logproba

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rs, ds, infos = self.env.step(torch.clamp(
                    action, self.env.action_space.low.mean(), self.env.action_space.high.mean()))
                # print(self.rewards[step])
                self.rewards[step], next_done = rs.view(
                    -1), torch.Tensor(ds).to(self.device)

                for info in infos:
                    if 'episode' in info.keys():
                        print(
                            f"global_step={global_step}, episode_reward={info['episode']['r']}")
                        self.writer.add_scalar("charts/episode_reward",
                                               info['episode']['r'], global_step)
                        break

            # bootstrap reward if not done. reached the batch limit
            with torch.no_grad():
                last_value = self.get_value(
                    next_obs.to(self.device)).reshape(1, -1)
                if self.args.gae:
                    advantages = torch.zeros_like(self.rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = last_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t+1]
                            nextvalues = self.values[t+1]
                        delta = self.rewards[t] + self.args.gamma * \
                            nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + self.args.gamma * \
                            self.args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + self.values
                else:
                    returns = torch.zeros_like(self.rewards).to(self.device)
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = last_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t+1]
                            next_return = returns[t+1]
                        returns[t] = self.rewards[t] + self.args.gamma * \
                            nextnonterminal * next_return
                    advantages = returns - self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,)+self.env.observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape(
                (-1,)+self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizaing the policy and value network
            target_agent = Agent(self.env, self.hp_dict).to(self.device)
            inds = np.arange(self.args.batch_size,)
            stopped = 0
            for i_epoch_pi in range(self.args.update_epochs):
                np.random.shuffle(inds)
                target_agent.load_state_dict(self.state_dict())
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    minibatch_ind = inds[start:end]
                    mb_advantages = b_advantages[minibatch_ind]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()
                                         ) / (mb_advantages.std() + 1e-8)

                    _, newlogproba, entropy = self.get_action(
                        b_obs[minibatch_ind], b_actions[minibatch_ind])
                    ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                    # Stats
                    approx_kl = (
                        b_logprobs[minibatch_ind] - newlogproba).mean()

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * \
                        torch.clamp(ratio, 1-self.args.clip_coef,
                                    1+self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()

                    # Value loss
                    new_values = self.get_value(b_obs[minibatch_ind]).view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (
                            (new_values - b_returns[minibatch_ind]) ** 2)
                        v_clipped = b_values[minibatch_ind] + torch.clamp(
                            new_values - b_values[minibatch_ind], -self.args.clip_coef, self.args.clip_coef)
                        v_loss_clipped = (
                            v_clipped - b_returns[minibatch_ind])**2
                        v_loss_max = torch.max(
                            v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * \
                            ((new_values -
                             b_returns[minibatch_ind]) ** 2).mean()

                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                if self.args.kle_stop:
                    if approx_kl > self.args.target_kl:
                        stopped += 1
                        break
                if self.args.kle_rollback:
                    if (b_logprobs[minibatch_ind] - self.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind])[1]).mean() > self.args.target_kl:
                        self.load_state_dict(target_agent.state_dict())
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes

            if (self.args.prod_mode):
                wandb.log({"charts/learning_rate": self.optimizer.param_groups[0]['lr'],
                           "losses/value_loss": v_loss.item(),
                           "losses/policy_loss": pg_loss.item(),
                           "losses/entropy": entropy.mean().item(),
                           "losses/approx_kl": approx_kl.item(),
                           "status/mean_returns": b_returns.mean(),
                           "status/early_stopped": stopped})
            print("SPS:", int(global_step / (time.time() - start_time)))

    def save_model(self, path="DRLPSO/models/"):
        torch.save(self.state_dict(), path+f'Agent__{s}')
