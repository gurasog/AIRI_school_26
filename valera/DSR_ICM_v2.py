import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torchsummary import summary
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange


def parse_args():
    
    class Args:
        
        def __init__(self):
            pass
        
    args_dict = dict(batch_size=64,
              buffer_size=1000000,
              capture_video=True,
              capture_video_frequency=50,
              cuda=True, end_e=0.02,
              env_id='ALE/Pong-v5',
              eta = 0.1,
              exp_name='DSR_ICM_v2',
              exploration_fraction=0.2,
              gamma=0.99, hf_entity='',
              learning_rate=0.0001,
              learning_starts=80000,
              num_envs=1,
              save_model=True,
              seed=1,
              start_e=1,
              target_network_frequency=5000,
              tau=1.0,
              torch_deterministic=True,
              total_timesteps=5000000,
              track=False, train_frequency=4,
              upload_model=False,
              wandb_entity=None, 
              wandb_project_name='cleanRL')
    args = Args()
    args.__dict__ = args_dict
    
    return args


def make_env(env_id, seed, idx, capture_video, run_name, capture_video_frequency):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger = lambda x: x % capture_video_frequency == 0)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk

class DeepSR_ICM(nn.Module):
    def __init__(self,
                 env,
                 hidden_dim = 256):
        super().__init__()
        self.n_actions = env.single_action_space.n
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 7, stride=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(64*4*4, hidden_dim),
#            nn.ReLU(),
#            nn.Linear(512, 512),
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim + self.n_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Softmax(dim = 1)
        )
        
        self.reward_estimate = nn.Linear(hidden_dim, 1, bias = False)
        self.w = self.reward_estimate.weight
        self.sr_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2 , hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim * self.n_actions),
                Rearrange('b (h a) -> b h a', h = hidden_dim,  a = self.n_actions),
            )
        
        self.train_output = namedtuple("TrainingOutput",
                  ("hidden_state",
                   "hidden_next_state",
                   "hidden_next_state_predicted",
                   "action",
                   "action_predicted",
                   "r_estimation",
                   "r_intrinsic",
                   "m",
                   "Q"))
        self.prediction_output = namedtuple("PredictionOutput",
                  ("hidden_state",
                   "hidden_next_state_predicted",
                   "action",
                   "action_predicted",
                   "r_estimation",
                   "m",
                   "Q"))
    
    def forward(self, *args, Q_only = False):
        '''
        Input: if training : x (current state), x_n (next state)
               if eval: x (current state)
               
        Output: if training : hidden_state, hidden_next_state, hidden_next_state_predicted,
                              action, action_predicted,
                              r_estimation, r_intrinsic, 
                              m, Q
                if eval: hidden_state, hidden_next_state_predicted,
                         action, action_predicted,
                         r_estimation,
                         m, Q
        '''
        if self.training:
            if len(args) != 3:
                raise Exception("In training phase model takes three arguments - current state, action,  next state.")
            x = args[0]
            action = args[1]
            x_n = args[2]
            
            hidden = self.encoder(x / 255.0)
            hidden_n = self.encoder(x_n / 255.0).detach()
            r = self.reward_estimate(hidden)
            m = self.sr_encoder(hidden.detach())
            Q = (m * self.w.unsqueeze(2)).sum(1)
            
            if not Q_only: 
                action = F.one_hot(action, num_classes = self.n_actions).squeeze(1) * 1.0
                action_pred = self.inverse_model(torch.concat([hidden, hidden_n], 1))
                hidden_n_pred = self.feed_forward(torch.concat([hidden, action], 1))
                r_intrinsic = F.mse_loss(hidden_n_pred, hidden_n, reduction="none").sum(1, keepdim=True)
                
            else:
                action, action_pred, hidden_n_pred, r_intrinsic = None, None, None, None

            return self.train_output(hidden, hidden_n, hidden_n_pred, 
                                     action, action_pred, 
                                     r, r_intrinsic, 
                                     m, Q)
        else:
            if len(args) != 1:
                   raise Exception("In prediction phase model takes one argument - current state.")
            x = args[0]
            hidden = self.encoder(x / 255.0)
            r = self.reward_estimate(hidden)
            m = self.sr_encoder(hidden.detach())
            Q = (m * self.w.unsqueeze(2)).sum(1)
            
            if not Q_only: 
                action = F.one_hot(torch.argmax(Q, dim=1), num_classes = self.n_actions)
                hidden_n_pred = self.feed_forward(torch.concat([hidden, action], 1))
                action_pred = self.inverse_model(torch.concat([hidden, hidden_n_pred], 1))
                
            else:
                action, hidden_n_pred, action_pred  = None, None, None
            
            return self.prediction_output(hidden, hidden_n_pred, 
                                       action, action_pred, 
                                       r,
                                       m, Q)
        
class SREncoder(nn.Module):
    
    def __init__(self,
                 env,
                 hidden_dim = 256):
        super().__init__()
        self.n_actions = env.single_action_space.n
        self.sr_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2 , hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim * self.n_actions),
                Rearrange('b (h a) -> b h a', h = hidden_dim,  a = self.n_actions),
            )
    
    def forward(self, hidden):
        m = self.sr_encoder(hidden.detach())
        return m
    

def exponential_schedule(start_e: float, end_e: float, duration: int, t: int):
    return max(start_e * (0.999997698) ** t, end_e)



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,  args.capture_video_frequency) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    sr_network = DeepSR_ICM(envs).to(device)
    optimizer = optim.AdamW(sr_network.parameters(), lr=args.learning_rate)
    target_network = SREncoder(envs).to(device)
    target_network.sr_encoder.load_state_dict(sr_network.sr_encoder.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = exponential_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            sr_network.eval()
            q_values = sr_network(torch.Tensor(obs).to(device), Q_only = True).Q
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    sr_network.eval()
                    output = sr_network(data.next_observations, Q_only = True)
                    m = target_network(output.hidden_state)
                    actions = output.Q.max(dim = 1).indices
                    m_next = m[torch.arange(actions.shape[0]),:,actions]
                    td_target = output.hidden_state.detach() + args.gamma * m_next * (1 - data.dones)
                sr_network.train()
                old_output = sr_network(data.observations, data.actions, data.next_observations)
                old_actions = data.actions.flatten()
                old_actions_ohe = F.one_hot(data.actions, num_classes = sr_network.n_actions).squeeze(1) * 1.0
                old_m = old_output.m[torch.arange(old_actions.shape[0]),:,old_actions]

                loss_m = F.mse_loss(td_target, old_m)           
                loss_r = F.mse_loss(data.rewards + args.eta*torch.sigmoid(old_output.r_intrinsic), old_output.r_estimation)
                loss_im = F.mse_loss(old_output.action_predicted, old_actions_ohe)
                loss_ff = F.mse_loss(old_output.hidden_next_state, old_output.hidden_next_state_predicted)
                loss = loss_m + loss_r + loss_ff + loss_im

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/loss_m", loss_m, global_step)
                    writer.add_scalar("losses/loss_r", loss_r, global_step)
                    writer.add_scalar("losses/loss_ff", loss_ff, global_step)
                    writer.add_scalar("losses/loss_im", loss_im, global_step)
                    writer.add_scalar("losses/q_values", old_output.Q.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network's sr_encoder
            if global_step % args.target_network_frequency == 0:
                for target_network_sr_param, sr_network_sr_param in zip(target_network.sr_encoder.parameters(), sr_network.sr_encoder.parameters()):
                    target_network_sr_param.data.copy_(
                        args.tau * sr_network_sr_param.data + (1.0 - args.tau) * target_network_sr_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(sr_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
    #    from cleanrl_utils.evals.dqn_eval import evaluate

    #    episodic_returns = evaluate(
    #        model_path,
    #        make_env,
    #        args.env_id,
    #        eval_episodes=10,
    #        run_name=f"{run_name}-eval",
    #        Model=DeepSR,
    #        device=device,
    #        epsilon=0.05,
    #    )
    #    for idx, episodic_return in enumerate(episodic_returns):
    #        writer.add_scalar("eval/episodic_return", episodic_return, idx)
    #
    #    if args.upload_model:
    #        from cleanrl_utils.huggingface import push_to_hub
    #
    #        repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #        push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")
    #
    envs.close()
    writer.close()