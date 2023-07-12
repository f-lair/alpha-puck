from math import ceil
from typing import Callable, Tuple

import numpy as np
import torch
from rl.critic import Critic, CriticDQN
from rl.replay_buffer import ReplayBuffer
from torch import nn, optim


class Agent:
    """Implements DecQN Agent."""

    def __init__(
        self,
        q_model: Critic | CriticDQN,
        discretization_dim: int,
        max_abs_force: float,
        max_abs_torque: float,
        device: torch.device,
        exploration_strategy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.q_model = q_model
        self.device = device
        self.double_q = isinstance(self.q_model, CriticDQN)
        self.exploration_strategy = exploration_strategy

        self.d2c_mapping = self._create_full_d2c_mapping(
            discretization_dim, max_abs_force, max_abs_torque
        ).to(
            self.device
        )

        if not self.double_q:
            # activate evaluation mode for inference model
            self.q_model.eval()

    @staticmethod
    def _select_action(critic_out: torch.Tensor) -> torch.Tensor:
        return torch.argmax(critic_out, dim=-1)

    @staticmethod
    def _eval_action(critic_out: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return critic_out.gather(dim=-1, index=action[..., None])[..., 0].mean(dim=-1)

    @staticmethod
    def _create_single_d2c_mapping(discretization_dim: int, max_abs_value: float) -> torch.Tensor:
        dv = 2 * max_abs_value / (discretization_dim - 1)
        return torch.arange(-max_abs_value, max_abs_value + dv, dv, dtype=torch.float32)

    @staticmethod
    def _create_full_d2c_mapping(
        discretization_dim: int, max_abs_force: float, max_abs_torque: float
    ) -> torch.Tensor:
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
        return self.d2c_mapping.gather(dim=-1, index=action[..., None])[..., 0]

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            if self.double_q:
                out, _ = self.q_model(torch.tensor(state, device=self.device, dtype=torch.float32))
            else:
                out = self.q_model(torch.tensor(state, device=self.device, dtype=torch.float32))

        action = self._select_action(out).squeeze()

        if self.double_q:
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
        assert self.double_q

        self.q_model.update_target_network()  # type: ignore

    def save_model(self, filepath: str) -> None:
        self.q_model.save(filepath)
