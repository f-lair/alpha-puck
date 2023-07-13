from math import ceil
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn, optim

from rl.critic import Critic, CriticDQN
from rl.replay_buffer import ReplayBuffer


class Agent:
    """Implements DecQN RL Agent."""

    def __init__(
        self,
        q_model: Critic | CriticDQN,
        discretization_dim: int,
        max_abs_force: float,
        max_abs_torque: float,
        device: torch.device,
        exploration_strategy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Initializes RL agent.

        Args:
            q_model (Critic | CriticDQN): Critic model.
            discretization_dim (int): Dimensionality of the action discretization.
            max_abs_force (float): Maximum absolute force used for translation.
            max_abs_torque (float): Maximum absolute torque used for rotation.
            device (torch.device): Device used for computations.
            exploration_strategy (Callable[[torch.Tensor], torch.Tensor] | None, optional): Strategy used for action space exploration. Defaults to None.
        """

        self.q_model = q_model
        self.device = device
        self.double_q = isinstance(self.q_model, CriticDQN)
        self.exploration_strategy = exploration_strategy

        self.d2c_mapping = self._create_full_d2c_mapping(
            discretization_dim, max_abs_force, max_abs_torque
        ).to(self.device)

        if not self.double_q:
            # activate evaluation mode for inference model
            self.q_model.eval()

    @staticmethod
    def _select_action(critic_out: torch.Tensor) -> torch.Tensor:
        """
        Selects action from output of the critic model.
        B: Batch dimension.
        A: Action dimension.
        D: Action discretization dimension.

        Args:
            critic_out (torch.Tensor): Critic model output [B, A, D].

        Returns:
            torch.Tensor: Selected action (discrete) [B, A].
        """

        return torch.argmax(critic_out, dim=-1)

    @staticmethod
    def _eval_action(critic_out: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluates action for a specific output of the critic model.
        B: Batch dimension.
        A: Action dimension.
        D: Action discretization dimension.

        Args:
            critic_out (torch.Tensor): Critic model output [B, A, D].
            action (torch.Tensor): Action (discrete) [B, A].

        Returns:
            torch.Tensor: Estimated Q-values [B].
        """

        return critic_out.gather(dim=-1, index=action[..., None])[..., 0].mean(dim=-1)

    @staticmethod
    def _create_single_d2c_mapping(discretization_dim: int, max_abs_value: float) -> torch.Tensor:
        """
        Creates single discrete-to-continuous mapping (for a single action dimension).
        D: Action discretization dimension.

        Args:
            discretization_dim (int): Dimensionality of the action discretization.
            max_abs_value (float): Maximum absolut value in continuous domain.

        Returns:
            torch.Tensor: Discrete-to-continous mapping [D].
        """

        dv = 2 * max_abs_value / (discretization_dim - 1)
        return torch.arange(-max_abs_value, max_abs_value + dv, dv, dtype=torch.float32)

    @staticmethod
    def _create_full_d2c_mapping(
        discretization_dim: int, max_abs_force: float, max_abs_torque: float
    ) -> torch.Tensor:
        """
        Creates full discrete-to-continuous mapping (over all action dimensions).
        A: Action dimension.
        D: Action discretization dimension.

        Args:
            discretization_dim (int): Dimensionality of the action discretization.
            max_abs_force (float): Maximum absolute force used for translation.
            max_abs_torque (float): Maximum absolute torque used for rotation.

        Returns:
            torch.Tensor: Discrete-to-continous mapping [A, D].
        """

        continuous_force = Agent._create_single_d2c_mapping(discretization_dim, max_abs_force)
        continuous_torque = Agent._create_single_d2c_mapping(discretization_dim, max_abs_torque)
        continuous_shoot = torch.concatenate(
            [
                torch.zeros((int(ceil(discretization_dim / 2)),), dtype=torch.float32),
                torch.ones((discretization_dim // 2,), dtype=torch.float32),
            ]
        )

        return torch.stack(
            [continuous_force, continuous_force, continuous_torque, continuous_shoot]
        )

    def _d2c(self, action: torch.Tensor) -> torch.Tensor:
        """
        Transforms discrete action into continuous action.
        A: Action dimension.

        Args:
            action (torch.Tensor): Action (discrete) [A].

        Returns:
            torch.Tensor: Action (continuous) [A].
        """

        return self.d2c_mapping.gather(dim=-1, index=action[..., None])[..., 0]

    def act(self, state: np.ndarray, eval_: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acts in environment, yielding an action, given a state.
        S: State dimension.
        A: Action dimension.

        Args:
            state (np.ndarray): State [S].
            eval_ (bool, optional): Whether evaluation mode is active. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action (continuous, discrete) [A].
        """

        with torch.no_grad():
            if self.double_q:
                out, _ = self.q_model(torch.tensor(state, device=self.device, dtype=torch.float32))
            else:
                out = self.q_model(torch.tensor(state, device=self.device, dtype=torch.float32))

        action = self._select_action(out).squeeze()

        if self.double_q and not eval_:
            assert self.exploration_strategy is not None
            action = self.exploration_strategy(action)

        return self._d2c(action).cpu().numpy(), action.cpu().numpy()

    def learn(
        self,
        replay_buffer: ReplayBuffer,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int,
        grad_clip_norm: float,
    ) -> float:
        """
        Performs a learning step.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer which stores past transitions.
            loss_fn (nn.Module): Loss function used for critic optimization.
            optimizer (optim.Optimizer): Optimizer used for critic optimization.
            batch_size (int): Batch size per learning step.
            grad_clip_norm (float): Maximum gradient norm above which gradients are clipped to.

        Returns:
            float: _description_
        """

        assert self.double_q

        (
            states,
            next_states,
            actions,
            rewards,
            terminals,
            weights,
            indices,
        ) = replay_buffer.sample(batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        terminals = terminals.to(self.device)
        weights = weights.to(self.device)

        optimizer.zero_grad()

        main_out, _ = self.q_model(states)
        q_estimate = self._eval_action(main_out, actions)

        # do not backpropagate through networks for target computation
        with torch.no_grad():
            main_out_next, target_out_next = self.q_model(next_states)
            target_actions = self._select_action(main_out_next)
            q_target = self._eval_action(target_out_next, target_actions)

        mask = 1 - terminals
        gamma = replay_buffer.gamma**replay_buffer.num_steps
        td_target = rewards + mask * gamma * q_target

        loss = loss_fn(q_estimate * weights, td_target * weights)

        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.q_model.main_parameters(), grad_clip_norm)  # type: ignore
        optimizer.step()

        td_delta = td_target - q_estimate
        replay_buffer.update_priorities(indices, torch.abs(td_delta.detach()).cpu())

        return loss.item()

    def update_target_network(self) -> None:
        """
        Updates target network.
        """

        assert self.double_q

        self.q_model.update_target_network()  # type: ignore

    def save_model(self, filepath: str) -> None:
        """
        Saves model checkpoint on disk.

        Args:
            filepath (str): Checkpoint filepath.
        """

        self.q_model.save(filepath)
