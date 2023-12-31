from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Implements multi-step Prioritized Experience Replay (PER) buffer."""

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
        device: torch.device,
    ) -> None:
        """
        Initializes replay buffer.

        Args:
            size (int): Size of the replay buffer.
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            num_steps (int): Number of steps in multi-step-return.
            min_priority (float): Minimum priority per transition in the replay buffer.
            alpha (float): Priority exponent in the replay buffer.
            beta (float): Importance sampling exponent in the replay buffer.
            gamma (float): Discount factor.
            device (torch.device): Device used for computations.
        """

        self.size = size
        self.num_steps = num_steps
        self.min_priority = min_priority
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device

        # memory buffers
        self.states = torch.empty((self.size, state_dim), device=self.device, dtype=torch.float32)
        self.next_states = torch.empty(
            (self.size, state_dim), device=self.device, dtype=torch.float32
        )
        self.actions = torch.empty((self.size, action_dim), device=self.device, dtype=torch.uint8)
        self.rewards = torch.empty((self.size,), device=self.device, dtype=torch.float32)
        self.terminals = torch.empty((self.size,), device=self.device, dtype=torch.uint8)
        self.counter = 0
        self.actual_size = 0

        # priority tree
        self.priority_tree = SumTree(size=self.size, data_size=1, data_type=torch.int64)
        self.max_priority = min_priority

        # multi-step buffer
        self.n_step_buffer = deque(maxlen=self.num_steps)

    def _get_multi_step(self) -> Tuple[float, torch.Tensor, bool]:
        """
        Returns transition information in multi-step return.
        S: State dimension.

        Returns:
            Tuple[float, torch.Tensor, bool]: Reward, next state [S], terminal flag.
        """

        _, next_state, _, reward, terminal = self.n_step_buffer[-1]

        # traverse collected transitions in reverse order, account for terminal signal by resetting
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
        """
        Stores transition in replay buffer.
        S: State dimension.
        A: Action dimension.

        Args:
            state (torch.Tensor): State [S].
            next_state (torch.Tensor): Next state [S].
            action (torch.Tensor): Action [A].
            reward (float): Reward.
            terminal (bool): Terminal flag.
        """

        state_t = torch.tensor(state, device=self.device)
        next_state_t = torch.tensor(next_state, device=self.device)
        action_t = torch.tensor(action, device=self.device)

        transition = (state_t, next_state_t, action_t, reward, terminal)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.num_steps:
            return

        reward, next_state_t, terminal = self._get_multi_step()
        state_t, _, action_t, _, _ = self.n_step_buffer[0]

        idx = self.counter
        self.counter = (self.counter + 1) % self.size
        self.actual_size = min(self.size, self.actual_size + 1)

        self.priority_tree.add(self.max_priority, torch.tensor(idx), idx)

        self.states[idx] = state_t
        self.next_states[idx] = next_state_t
        self.actions[idx] = action_t
        self.rewards[idx] = reward
        self.terminals[idx] = terminal

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
        """
        Samples a batch of transitions from replay buffer, according to their priority distribution.
        B: Batch dimension.
        S: State dimension.
        A: Action dimension.

        Args:
            sample_size (int): Number of transitions to be sampled (batch size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: States [B, S], next states [B, S], actions [B, A], rewards [B], terminals [B], importance sampling weights [B], buffer indices [B].
        """

        assert (
            self.actual_size >= sample_size
        ), "Replay buffer contains less samples than sample size!"

        transform_uniform = (
            lambda uniform_sample, lower, upper: (lower - upper) * uniform_sample + upper
        )

        segment = self.priority_tree.total / sample_size
        lower = segment * torch.arange(sample_size)
        upper = segment * torch.arange(1, sample_size + 1)

        cumsum = transform_uniform(torch.rand(sample_size, dtype=torch.float64), lower, upper)
        torch.clamp_(
            cumsum, min=0.0, max=self.priority_tree.total
        )  # counter numerical instabilities
        data_indices, priorities, sample_indices = self.priority_tree.get(cumsum)
        sample_indices = sample_indices[:, 0]

        probabilities = priorities / self.priority_tree.total
        probabilities[
            probabilities == 0.0
        ] = torch.inf  # weight = 0 in case of numerical instabilities

        weights = (self.actual_size * probabilities) ** (-self.beta)
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
            weights.to(device=self.device, dtype=torch.float32),
            data_indices,
        )

    def update_priorities(self, data_indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """
        Updates priority distribution at given indices in the replay buffer.
        B: Batch dimension.

        Args:
            data_indices (torch.Tensor): Indices in buffer [B].
            priorities (torch.Tensor): New priorities [B].
        """

        priorities = (priorities + self.min_priority) ** self.alpha

        assert self.priority_tree.nodes.min() >= 0

        self.priority_tree.update(data_indices, priorities)
        self.max_priority = max(self.max_priority, priorities.max().item())


class SumTree:
    """Implements SumTree for efficient computation of (cumulative) sums in a vectorized manner."""

    def __init__(self, size: int, data_size: int, data_type: torch.dtype) -> None:
        """
        Initializes sum tree.

        Args:
            size (int): Number of leaf nodes.
            data_size (int): Vector size per leaf node.
            data_type (torch.dtype): Vector data type of leaf nodes.
        """

        self.nodes = torch.zeros((2 * size - 1,), dtype=torch.float64)
        self.data = torch.zeros((size, data_size), dtype=data_type)

        self.size = size

    @property
    def total(self) -> float:
        """
        Returns value at root node (=sum of all leaf node values).

        Returns:
            float: Leaf node value.
        """

        return self.nodes[0].item()

    def update(self, data_indices: torch.Tensor, values: torch.Tensor) -> None:
        """
        Updates sum tree values bottom-up.
        B: Batch dimension.

        Args:
            data_indices (torch.Tensor): Leaf node indices, at which new values are inserted [B].
            values (torch.Tensor): New leaf node values [B].
        """

        compute_parent_indices = lambda indices: (indices - 1) // 2
        compute_mask = lambda parents: parents >= 0

        # NOTE: data_indices are always sorted due to arange-based segment creation in sample()
        indices_unique, unique_inv = torch.unique_consecutive(data_indices, return_inverse=True)
        values_unique = torch.zeros(indices_unique.shape, dtype=values.dtype)
        values_unique[unique_inv] = values

        tree_indices = indices_unique + self.size - 1
        changes = values_unique - self.nodes[tree_indices]

        self.nodes[tree_indices] = values_unique

        parent_indices = compute_parent_indices(tree_indices)
        mask = compute_mask(parent_indices)
        while torch.any(mask):
            parent_indices_unique, unique_inv = torch.unique_consecutive(
                parent_indices[mask], return_inverse=True
            )
            nodes_add = torch.zeros_like(self.nodes[parent_indices_unique])
            nodes_add.put_(unique_inv, changes[mask], accumulate=True)
            self.nodes[parent_indices_unique] += nodes_add

            parent_indices = compute_parent_indices(parent_indices)
            mask = compute_mask(parent_indices)

    def add(self, value: float, data: torch.Tensor, idx: int):
        """
        Adds new leaf node entry.

        Args:
            value (float): Leaf node value.
            data (torch.Tensor): Leaf node data vector.
            idx (int): Leaf node index.
        """

        self.data[idx] = data
        self.update(
            torch.tensor([idx]),
            torch.tensor([value], dtype=torch.float64),
        )

    def get(self, cumsum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns leaf nodes corresponding to given cumulative sums.
        B: Batch dimension.
        V: Data vector dimension.

        Args:
            cumsum (torch.Tensor): Cumulative sums [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Leaf node indices [B], corresponding values [B] and data vectors [B, V].
        """

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

    def get_leaves(self, data_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns leaf nodes corresponding to their indices.
        B: Batch dimension.
        V: Data vector dimension.

        Args:
            data_indices (torch.Tensor): Leaf node indices [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Corresponding leaf values [B] and data vectors [B, V].
        """

        tree_indices = data_indices + self.size - 1

        return self.nodes[tree_indices], self.data[data_indices]
