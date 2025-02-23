import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    """
    Example diffusion model that can produce next state deltas or
    full next states from the current state + action. This class
    can be trained offline on trajectory data, then loaded at runtime.
    """

    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        """
        :param input_dim: dimension of [state + action + optional flags]
        :param hidden_dim: hidden layer dimension
        :param output_dim: dimension of the delta for the next state (or full next state)
        """
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        return self.net(x)

    def sample_next_state(self, current_state_action):
        """
        Given the current state and action (and optional placeholders),
        predict a 'delta' or next state. For simplicity, we predict a
        small 4D delta: [dx, dy, dtheta, dv].

        :param current_state_action: (tensor) shape [batch, input_dim]
        :return: (tensor) shape [batch, output_dim]
        """
        with torch.no_grad():
            # forward pass
            delta = self.forward(current_state_action)
        return delta
