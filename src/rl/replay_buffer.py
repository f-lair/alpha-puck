from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Implements multi-step Prioritized Experience Replay (PER) buffer."""

    # cf. https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/14/prioritized-experience-replay.html
    # cf. https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
    # cf. https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/07.n_step_learning.ipynb

    def __init__(
        self,
        size: int,
        state_dim: int,
        action_dim: int,
        num_steps: int,
        min_priority: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> None:
        self.size = size
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_priority = min_priority

        # memory buffers
        self.states = torch.empty((self.size, state_dim), dtype=torch.float32)
        self.next_states = torch.empty((self.size, state_dim), dtype=torch.float32)
        self.actions = torch.empty((self.size, action_dim), dtype=torch.uint8)
        self.rewards = torch.empty((self.size,), dtype=torch.float32)
        self.terminals = torch.empty((self.size,), dtype=torch.uint8)
        self.counter = 0

        # priority tree
        self.priority_tree = SumTree(size=self.size, data_size=1, data_type=torch.int64)
        self.max_priority = min_priority

        # multi-step buffer
        self.n_step_buffer = deque(maxlen=self.num_steps)

    def is_empty(self) -> bool:
        return self.counter == 0

    def is_full(self) -> bool:
        return self.counter == self.size - 1

    def _store(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        terminal: bool,
        idx: int | None = None,
    ) -> None:
        if idx is None:
            idx = self.counter
            self.counter += 1

        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminals[idx] = terminal

    def _get_step_info(self) -> Tuple[float, torch.Tensor, bool]:
        _, next_state, _, reward, terminal = self.n_step_buffer[-1]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            _, next_state_tm1, _, reward_tm1, terminal_tm1 = transition

            reward = reward_tm1 + self.gamma * reward * (1 - terminal_tm1)
            if terminal_tm1:
                next_state, terminal = next_state_tm1, terminal_tm1

        return reward, next_state, terminal

    def store(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
    ) -> None:
        state_t = torch.tensor(state)
        next_state_t = torch.tensor(next_state)
        action_t = torch.tensor(action)

        transition = (state_t, next_state_t, action_t, reward, terminal)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.num_steps:
            return

        reward, next_state_t, terminal = self._get_step_info()
        state_t, _, action_t, _, _ = self.n_step_buffer[0]

        data_idx = self.priority_tree.minimum[2] if self.is_full() else None

        self.priority_tree.add(self.max_priority, torch.tensor(self.counter), data_idx)
        self._store(state_t, next_state_t, action_t, reward, terminal)

    def sample(
        self, sample_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self.counter >= sample_size, "Replay buffer contains less samples than sample size!"

        transform_uniform = (
            lambda uniform_sample, lower, upper: (lower - upper) * uniform_sample + upper
        )

        segment = self.priority_tree.total / sample_size
        lower = segment * torch.arange(sample_size)
        upper = segment * torch.arange(1, sample_size + 1)

        cumsum = transform_uniform(torch.rand(sample_size, dtype=torch.float32), lower, upper)
        data_indices, priorities, sample_indices = self.priority_tree.get(cumsum)
        sample_indices = sample_indices[:, 0]

        probabilities = priorities / self.priority_tree.total
        probabilities[
            probabilities == 0.0
        ] = torch.inf  # weight = 0 in case of numerical instabilities

        weights = (self.counter * probabilities) ** (-self.beta)
        weights /= weights.max()

        states = self.states[sample_indices]
        next_states = self.next_states[sample_indices]
        actions = self.actions[sample_indices].to(torch.int64)
        rewards = self.rewards[sample_indices]
        terminals = self.terminals[sample_indices].to(torch.float32)

        return (
            states,
            next_states,
            actions,
            rewards,
            terminals,
            weights.to(torch.float32),
            data_indices,
        )

    def update_priorities(self, data_indices: torch.Tensor, priorities: torch.Tensor) -> None:
        priorities = (priorities + self.min_priority) ** self.alpha
        assert self.priority_tree.nodes.min() >= 0
        self.priority_tree.update(data_indices, priorities)
        self.max_priority = max(self.max_priority, priorities.max().item())


class SumTree:
    """Implements SumTree for efficient computation of (cumulative) sums."""

    # cf. https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py

    def __init__(self, size: int, data_size: int, data_type: torch.dtype) -> None:
        self.nodes = torch.zeros((2 * size - 1,), dtype=torch.float32)
        self.data = torch.zeros((size, data_size), dtype=data_type)

        self.size = size
        self.counter = 0
        self.minimum_idx = self.size - 1

    @property
    def total(self) -> float:
        return self.nodes[0].item()

    @property
    def minimum(self) -> Tuple[float, torch.Tensor, int]:
        data_idx = self.minimum_idx - self.size + 1
        return self.nodes[self.minimum_idx].item(), self.data[data_idx], data_idx

    def update(self, data_indices: torch.Tensor, values: torch.Tensor) -> None:
        compute_parent_indices = lambda indices: (indices - 1) // 2
        compute_mask = lambda parents: parents >= 0

        # NOTE: data_indices are always sorted due to arange-based segment creation in sample()
        indices_unique, unique_inv = torch.unique_consecutive(data_indices, return_inverse=True)
        values_unique = torch.zeros(indices_unique.shape, dtype=values.dtype)
        values_unique[unique_inv] = values

        tree_indices = indices_unique + self.size - 1
        changes = values_unique - self.nodes[tree_indices]

        # print("O", self.nodes[tree_indices])
        # print("D", data_indices)
        # print("V", values)
        # print("C", changes)

        self.nodes[tree_indices] = values_unique

        parent_indices = compute_parent_indices(tree_indices)
        mask = compute_mask(parent_indices)
        while torch.any(mask):
            parent_indices_unique, unique_inv = torch.unique_consecutive(
                parent_indices[mask], return_inverse=True
            )
            nodes_add = torch.zeros_like(self.nodes[parent_indices_unique])
            nodes_add.put_(unique_inv, changes[mask], accumulate=True)
            # if not torch.all(self.nodes[parent_indices_unique] + nodes_add >= 0):
            #     print("OP", self.nodes[parent_indices_unique])
            #     print("A", nodes_add)
            #     print("S", self.nodes[parent_indices_unique] + nodes_add)
            #     print("U", unique_inv)
            #     print("PI", parent_indices[mask])
            self.nodes[parent_indices_unique] += nodes_add

            parent_indices = compute_parent_indices(parent_indices)
            mask = compute_mask(parent_indices)

        min_value_idx = torch.argmin(values_unique)

        if values[min_value_idx] < self.nodes[self.minimum_idx]:
            self.minimum_idx = int(tree_indices[min_value_idx].item())

    def add(self, value: float, data: torch.Tensor, idx: int | None = None):
        if idx is None:
            idx = self.counter
            self.counter += 1

        self.data[idx] = data
        self.update(torch.tensor([idx]), torch.tensor([value], dtype=torch.float32))

    def get(self, cumsum: torch.Tensor):
        assert torch.all(cumsum <= self.total), f"{cumsum} > {self.total}"

        compute_mask = lambda indices: 2 * indices + 1 < len(self.nodes)

        tree_indices = torch.zeros_like(cumsum, dtype=torch.int64)
        mask = compute_mask(tree_indices)

        while torch.any(mask):
            left = 2 * tree_indices + 1
            right = left + 1

            left_mask = torch.zeros_like(cumsum, dtype=torch.bool)
            left_mask[mask] = cumsum[mask] <= self.nodes[left[mask]]

            right_mask = torch.zeros_like(cumsum, dtype=torch.bool)
            right_mask[mask] = cumsum[mask] > self.nodes[left[mask]]

            tree_indices[left_mask] = left[left_mask]
            tree_indices[right_mask] = right[right_mask]

            cumsum[right_mask] -= self.nodes[left[right_mask]]

            mask = compute_mask(tree_indices)

        data_indices = tree_indices - self.size + 1

        return data_indices, self.nodes[tree_indices], self.data[data_indices]
