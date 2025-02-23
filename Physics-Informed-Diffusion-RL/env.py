import gym
import numpy as np
from gym import spaces

import torch

class KinematicBicycleEnv(gym.Env):
    """
    Kinematic Bicycle Model environment with optional diffusion-based stochastic transitions.
    State space (x, y, theta, v, error_x, error_y).
    Action space (acceleration, steering rate).

    If diffusion_model is provided, the environment transitions incorporate
    samples from the diffusion model. This allows more realistic or learned
    stochastic transitions. Otherwise, it uses standard kinematic updates + random noise.
    """

    def __init__(
        self,
        diffusion_model=None,
        dt=0.1,
        L=2.5,
        max_speed=30.0,
        max_steering=0.5,
        accel_bounds=(-3.0, 3.0),
        steer_rate_bounds=(-0.5, 0.5),
        seed=42
    ):
        super(KinematicBicycleEnv, self).__init__()

        self.diffusion_model = diffusion_model  # optional
        self.dt = dt
        self.L = L
        self.max_speed = max_speed
        self.max_steering = max_steering

        self.action_space = spaces.Box(
            low=np.array([accel_bounds[0], steer_rate_bounds[0]], dtype=np.float32),
            high=np.array([accel_bounds[1], steer_rate_bounds[1]], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        # State: [x, y, theta, v, error_x, error_y]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)

        self.reset()

    def step(self, action):
        accel, delta_rate = action
        # Clip controls
        accel = np.clip(accel, self.action_space.low[0], self.action_space.high[0])
        delta_rate = np.clip(delta_rate, self.action_space.low[1], self.action_space.high[1])

        # Nominal kinematic update
        new_delta = np.clip(self.delta + delta_rate * self.dt, -self.max_steering, self.max_steering)
        new_v = np.clip(self.v + accel * self.dt, 0, self.max_speed)
        new_x = self.x + new_v * np.cos(self.theta) * self.dt
        new_y = self.y + new_v * np.sin(self.theta) * self.dt
        new_theta = self.theta + (new_v / self.L) * np.tan(new_delta) * self.dt

        # If we have a diffusion model, we optionally adjust the transition
        if self.diffusion_model is not None:
            # Construct a tensor for the current state + action
            # shape: (batch=1, features=8) -> [x, y, theta, v, delta, accel, delta_rate, 0?]
            # The "0" or some placeholder can be used if the diffusion model expects a certain input size
            input_vec = torch.tensor(
                [[self.x, self.y, self.theta, self.v, self.delta, accel, delta_rate, 0.0]],
                dtype=torch.float32
            )
            # Sample next state delta from the diffusion model
            delta_state = self.diffusion_model.sample_next_state(input_vec)
            # Adjust new state by adding delta or overriding (depending on design)
            # Here we do a simple additive adjustment for demonstration
            new_x += float(delta_state[0, 0])
            new_y += float(delta_state[0, 1])
            new_theta += float(delta_state[0, 2])
            new_v += float(delta_state[0, 3])

        # Update environment state
        self.x, self.y, self.theta, self.v, self.delta = (
            new_x, new_y, new_theta, new_v, new_delta
        )

        # Simulated deception or measurement errors
        if self.rng.random() < 0.1:
            self.error_x = self.rng.uniform(-2.0, 2.0)
        else:
            self.error_x = 0.0
        if self.rng.random() < 0.1:
            self.error_y = self.rng.uniform(-2.0, 2.0)
        else:
            self.error_y = 0.0

        # Compute reward
        # (example: penalize large errors + large control inputs)
        reward = - (
            (self.error_x**2 + self.error_y**2)
            + 0.1 * (accel**2 + delta_rate**2)
        )

        # For demonstration, we don't define a terminal condition
        done = False
        # Create observation
        obs = np.array([
            self.x,
            self.y,
            self.theta,
            self.v,
            self.error_x,
            self.error_y
        ], dtype=np.float32)
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 5.0
        self.delta = 0.0
        self.error_x = 0.0
        self.error_y = 0.0
        return np.array(
            [self.x, self.y, self.theta, self.v, self.error_x, self.error_y],
            dtype=np.float32
        )
