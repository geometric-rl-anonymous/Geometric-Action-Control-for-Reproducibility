#!/usr/bin/env python3

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import sys, os, glob
import ctypes
from dataclasses import dataclass
from tqdm import tqdm
import json


@dataclass
class Args:
    test: float = 10


def _load_gac():
    import torch
    import ctypes
    import sys
    import os
    import importlib.util


    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    libc10 = os.path.join(torch_lib, 'libc10.so')
    if os.path.exists(libc10):
        ctypes.CDLL(libc10, mode=ctypes.RTLD_GLOBAL)


    if not os.path.exists("gac.so"):
        raise RuntimeError("❌ Cannot find gac.so")

    spec = importlib.util.spec_from_file_location("gac_core", "gac.so")
    gac_core = importlib.util.module_from_spec(spec)
    sys.modules["gac_core"] = gac_core
    sys.modules["gac"] = gac_core
    spec.loader.exec_module(gac_core)
    print("✅ GAC loaded successfully")
    return gac_core


gac_core = _load_gac()

class GAC_Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.args = args

        # Network structure (only used to load weights)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.kappa_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_params_dict(self):

        return {
            'backbone_weight1': self.backbone[0].weight,
            'backbone_bias1': self.backbone[0].bias,
            'backbone_weight2': self.backbone[2].weight,
            'backbone_bias2': self.backbone[2].bias,
            'mu_head_weight': self.mu_head.weight,
            'mu_head_bias': self.mu_head.bias,
            'kappa_head_weight1': self.kappa_head[0].weight,
            'kappa_head_bias1': self.kappa_head[0].bias,
            'kappa_head_weight2': self.kappa_head[2].weight,
            'kappa_head_bias2': self.kappa_head[2].bias,
        }


    def get_action(self, x, deterministic=False):
        params = self.get_params_dict()

        action, _, _ = gac_core.get_action_with_entropy(
            x,
            **params,
            kappa_init=self.args.test,
            action_scale=self.action_scale,
            action_bias=self.action_bias
        )
        # Auto-detect Ant environment
        if 'Ant' in self.args.env_id:
            return action / 2.5  # r=1.0 for Ant
        return action  # r=2.5 for others
        # TODO: When evaluating Ant checkpoints, rescale actions by 1/2.5 to match r = 1.0.



def evaluate_gac(checkpoint_path, num_episodes=10, deterministic=True):

    print("=" * 60)
    print("GAC Evaluation ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', Args())
    global_step = checkpoint.get('global_step', 0)

    print(f"📁 model: {os.path.basename(checkpoint_path)}")
    print(f"🔢 steps: {global_step:,}")
    print(f"💻 device: {device}")
    print(f"🎮 env: {args.env_id}")

    # 创建环境
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # create Actor
    actor = GAC_Actor(obs_dim, action_dim, args).to(device)

    # load weight
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k not in ['action_scale', 'action_bias']}
    actor.load_state_dict(filtered_state_dict, strict=False)

    # set action scale/bias
    if 'action_scale' in state_dict:
        actor.action_scale = state_dict['action_scale'].to(device)
    else:
        actor.action_scale = torch.tensor(
            (env.action_space.high - env.action_space.low) / 2.0,
            dtype=torch.float32
        ).to(device)

    if 'action_bias' in state_dict:
        actor.action_bias = state_dict['action_bias'].to(device)
    else:
        actor.action_bias = torch.tensor(
            (env.action_space.high + env.action_space.low) / 2.0,
            dtype=torch.float32
        ).to(device)

    actor.eval()

    # eval
    print(f"\n📊 eval {num_episodes} episodes ({'deterministic' if deterministic else 'stochastic'})")
    print("-" * 40)

    episode_returns = []

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=episode_idx)
        episode_return = 0
        episode_length = 0

        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action = actor.get_action(obs_tensor, deterministic=deterministic)
                action_np = action.cpu().numpy().flatten()
                action_np = np.clip(action_np, -1, 1)

            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_return += reward
            episode_length += 1

            if terminated or truncated:
                if "episode" in info:
                    true_return = info["episode"]["r"]
                    if isinstance(true_return, np.ndarray):
                        true_return = true_return.item()
                    episode_returns.append(true_return)
                    print(f"Episode {episode_idx + 1:2d}: {true_return:8.2f}")
                break

    env.close()

    # statistics
    print("-" * 40)
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    print(f"📈 avg.: {mean_return:.2f} ± {std_return:.2f}")
    print(f"📈 max.: {np.max(episode_returns):.2f}")
    print(f"📈 min.: {np.min(episode_returns):.2f}")
    print("=" * 60)


    results = {
        'checkpoint': os.path.basename(checkpoint_path),
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'episodes': episode_returns
    }

    # result_file = checkpoint_path.replace('.pt', '_eval.json')
    # with open(result_file, 'w') as f:
    #     json.dump(results, f, indent=2)
    # print(f"💾 Save the results: {os.path.basename(result_file)}")

    return mean_return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--stochastic', action='store_true')

    args = parser.parse_args()

    if args.checkpoint is None:
        # choose envs
        # checkpoints = glob.glob("gac_pretrain_model/HalfCheetah-v4_step_500000.pt")
        # checkpoints = glob.glob("gac_pretrain_model/Humanoid-v4_step_1000000.pt")
        # checkpoints = glob.glob("gac_pretrain_model/Ant-v4_step_500000.pt")
        checkpoints = glob.glob("gac_pretrain_model/Walker2d-v4_step_1000000.pt")

        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = checkpoints[-1]
        else:
            print("❌ cannot find checkpoint")
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint

    deterministic = not args.stochastic
    evaluate_gac(checkpoint_path, args.episodes, deterministic)


if __name__ == "__main__":
    main()