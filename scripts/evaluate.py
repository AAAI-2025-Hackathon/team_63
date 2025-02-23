import argparse
import os
import numpy as np

import torch

from rlpidpm import (
    KinematicBicycleEnv,
    create_agent,
    DiffusionModel,
    load_model,
    Logger,
    set_random_seed
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent with diffusion environment.")
    parser.add_argument("--algo", type=str, default="PPO", help="RL algorithm: PPO, DQN, or SAC")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved RL agent (stable-baselines3 format).")
    parser.add_argument("--diffusion_model_path", type=str, default=None,
                        help="Path to a trained diffusion model (PyTorch state_dict). If None, no diffusion used.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_path", type=str, default="logs/eval_log.txt")
    args = parser.parse_args()

    set_random_seed(args.seed)
    logger = Logger(log_path=args.log_path)

    # Load diffusion model if available
    diffusion_model = None
    if args.diffusion_model_path is not None and os.path.exists(args.diffusion_model_path):
        diffusion_model_instance = DiffusionModel()
        diffusion_model_instance.load_state_dict(torch.load(args.diffusion_model_path))
        diffusion_model_instance.eval()
        diffusion_model = diffusion_model_instance
        logger.log(f"Loaded diffusion model from {args.diffusion_model_path}")

    # Create environment
    env = KinematicBicycleEnv(diffusion_model=diffusion_model, seed=args.seed)

    # Load RL model
    # We pass create_agent's class but we call load_model
    # Because for stable-baselines3, we can do [AgentClass].load(path)
    # So we do something like:
    if args.algo.upper() == "PPO":
        from stable_baselines3 import PPO as AgentClass
    elif args.algo.upper() == "DQN":
        from stable_baselines3 import DQN as AgentClass
    elif args.algo.upper() == "SAC":
        from stable_baselines3 import SAC as AgentClass
    else:
        raise ValueError(f"Unsupported algo: {args.algo}")

    agent = load_model(AgentClass, args.model_path)

    # Evaluate
    rewards_list = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        rewards_list.append(ep_reward)
        logger.log(f"Episode {ep+1}, Reward={ep_reward:.3f}")

    avg_reward = np.mean(rewards_list)
    logger.log(f"Evaluation complete. Average reward over {args.episodes} episodes: {avg_reward:.3f}")

if __name__ == "__main__":
    main()
