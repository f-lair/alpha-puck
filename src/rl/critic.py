from typing import Iterator, Tuple

import torch
from torch import nn


class Critic(nn.Module):
    """Implements single critic network."""

    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, discretization_dim: int
    ) -> None:
        """
        Initializes critic network.

        Args:
            state_dim (int): Dimensionality of the state space.
            hidden_dim (int): Dimensionality of the hidden layers in the critic model.
            action_dim (int): Dimensionality of the action space.
            discretization_dim (int): Dimensionality of the action discretization.
        """

        super().__init__()

        self.action_dim = action_dim
        self.discretization_dim = discretization_dim

        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, action_dim * discretization_dim)

        self.act = nn.ELU()  # ELU instead of ReLU (?)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass for a given input state.
        B: Batch dimension.
        S: State dimension.
        A: Action dimension.
        D: Action discretization dimension.

        Args:
            state (torch.Tensor): State [B, S]

        Returns:
            torch.Tensor: Critic model output [B, A, D].
        """

        out1 = self.lin1(state)  # no activation after first fc layer (?)
        out = self.lin2(out1)
        out += out1  # residual connection
        out = self.layer_norm(out)
        out = self.act(out)
        out = self.lin3(out)

        return out.view(-1, self.action_dim, self.discretization_dim)

    def save(self, filepath: str) -> None:
        """
        Saves model checkpoint on disk.

        Args:
            filepath (str): Checkpoint filepath.
        """

        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """
        Loads model checkpoint from disk.

        Args:
            filepath (str): Checkpoint filepath.
        """

        self.load_state_dict(torch.load(filepath))


class CriticDQN(nn.Module):
    """Implements two Critic networks for Double-Q-Learning."""

    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, discretization_dim: int
    ) -> None:
        """
        Initializes main and target critic networks.

        Args:
            state_dim (int): Dimensionality of the state space.
            hidden_dim (int): Dimensionality of the hidden layers in the critic model.
            action_dim (int): Dimensionality of the action space.
            discretization_dim (int): Dimensionality of the action discretization.
        """

        super().__init__()

        self.main_critic = Critic(state_dim, hidden_dim, action_dim, discretization_dim)
        self.target_critic = Critic(state_dim, hidden_dim, action_dim, discretization_dim)

        # synchronize initially
        self.target_critic.load_state_dict(self.main_critic.state_dict())

        # activate evaluation mode for target model
        self.target_critic.eval()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass for a given input state.
        B: Batch dimension.
        S: State dimension.
        A: Action dimension.
        D: Action discretization dimension.

        Args:
            state (torch.Tensor): State [B, S]

        Returns:
            torch.Tensor: Critic model output [B, A, D].
        """

        main_out = self.main_critic(state)
        target_out = self.target_critic(state)

        return main_out, target_out

    def update_target_network(self) -> None:
        """
        Updates target network by copying parameters from main network.
        """

        main_params = dict(self.main_critic.named_parameters())
        target_params = dict(self.target_critic.named_parameters())

        # do not create gradients for params update
        with torch.no_grad():
            for key in main_params:
                target_params[key].copy_(main_params[key])

    def main_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """
        Returns parameters of main network.

        Yields:
            Iterator[nn.parameter.Parameter]: Main network parameters.
        """

        return self.main_critic.parameters()

    def save(self, filepath: str) -> None:
        """
        Saves model checkpoint of main network on disk.

        Args:
            filepath (str): Checkpoint filepath.
        """

        self.main_critic.save(filepath)

    def load(self, filepath: str) -> None:
        """
        Loads model checkpoint for main and target networks from disk.

        Args:
            filepath (str): Checkpoint filepath.
        """

        self.main_critic.load(filepath)
        self.target_critic.load_state_dict(self.main_critic.state_dict())
