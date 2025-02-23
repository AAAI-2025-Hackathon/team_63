import torch
from stable_baselines3 import PPO, DQN, SAC

def create_agent(
    algo_name,
    env,
    learning_rate=1e-3,
    seed=42,
    **kwargs
):
    """
    Creates a stable-baselines3 RL agent with the specified algorithm name.
    Supported algo_name: 'PPO', 'DQN', 'SAC'.

    :param algo_name: (str) Which RL algorithm to instantiate.
    :param env: (gym.Env) The environment to train/evaluate on.
    :param learning_rate: (float) Learning rate for the agent.
    :param seed: (int) Random seed.
    :param kwargs: Additional arguments for the specific algorithm (e.g. hyperparameters).
    :return: A stable-baselines3 RL model.
    """
    torch.manual_seed(seed)

    if algo_name.upper() == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs
        )
    elif algo_name.upper() == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs
        )
    elif algo_name.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported algo_name: {algo_name}")
    
    return model
