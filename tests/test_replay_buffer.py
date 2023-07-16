import numpy as np
import pytest
import torch
from torch.testing import assert_close

from src.rl.replay_buffer import ReplayBuffer

N = 6


def test_store():
    num_steps = 2
    min_prio = 1e-2
    gamma = 0.5
    replay_buffer = ReplayBuffer(
        N - num_steps + 1, 1, 1, num_steps, min_prio, 0, 1.0, 0.0, gamma, 0.0, 0.0
    )

    actions = np.arange(N, dtype=int)
    states = 0.1 * actions
    next_states = -states

    rewards = 10.0 * actions
    terminals = [False, False, True, False, False, True]

    replay_buffer.store(states[0], next_states[0], actions[0], rewards[0], terminals[0])
    assert replay_buffer.actual_size == 0
    assert replay_buffer.counter == 0
    replay_buffer.store(states[1], next_states[1], actions[1], rewards[1], terminals[1])
    assert replay_buffer.actual_size == 1
    assert replay_buffer.counter == 1
    assert replay_buffer.states[0] == states[0]
    assert replay_buffer.next_states[0] == next_states[1]
    assert replay_buffer.actions[0] == actions[0]
    assert replay_buffer.rewards[0] == rewards[0] + gamma * rewards[1]
    assert replay_buffer.terminals[0] == terminals[1]
    replay_buffer.store(states[2], next_states[2], actions[2], rewards[2], terminals[2])
    assert replay_buffer.actual_size == 2
    assert replay_buffer.counter == 2
    assert replay_buffer.states[1] == states[1]
    assert replay_buffer.next_states[1] == next_states[2]
    assert replay_buffer.actions[1] == actions[1]
    assert replay_buffer.rewards[1] == rewards[1] + gamma * rewards[2]
    assert replay_buffer.terminals[1] == terminals[2]
    replay_buffer.store(states[3], next_states[3], actions[3], rewards[3], terminals[3])
    assert replay_buffer.actual_size == 3
    assert replay_buffer.counter == 3
    assert replay_buffer.states[2] == states[2]
    assert replay_buffer.next_states[2] == next_states[2]
    assert replay_buffer.actions[2] == actions[2]
    assert replay_buffer.rewards[2] == rewards[2]
    assert replay_buffer.terminals[2] == terminals[2]
    replay_buffer.store(states[4], next_states[4], actions[4], rewards[4], terminals[4])
    assert replay_buffer.actual_size == 4
    assert replay_buffer.counter == 4
    assert replay_buffer.states[3] == states[3]
    assert replay_buffer.next_states[3] == next_states[4]
    assert replay_buffer.actions[3] == actions[3]
    assert replay_buffer.rewards[3] == rewards[3] + gamma * rewards[4]
    assert replay_buffer.terminals[3] == terminals[4]
    replay_buffer.store(states[5], next_states[5], actions[5], rewards[5], terminals[5])
    assert replay_buffer.actual_size == 5
    assert replay_buffer.counter == 0
    assert replay_buffer.states[4] == states[4]
    assert replay_buffer.next_states[4] == next_states[5]
    assert replay_buffer.actions[4] == actions[4]
    assert replay_buffer.rewards[4] == rewards[4] + gamma * rewards[5]
    assert replay_buffer.terminals[4] == terminals[5]


def test_update_priorities():
    num_steps = 1
    min_prio = 1e-2
    gamma = 1.0
    replay_buffer = ReplayBuffer(N, 1, 1, num_steps, min_prio, 0, 1.0, 0.0, gamma, 0.0, 0.0)

    data_indices = torch.tensor([1, 3, 5], dtype=torch.int64)
    priorities = torch.tensor([0.0, 2e-2, 4e-2])

    replay_buffer.update_priorities(data_indices, priorities)
    assert replay_buffer.max_priority == pytest.approx(5e-2)
    assert_close(replay_buffer.priority_tree.nodes[data_indices + N - 1], priorities + min_prio)

    replay_buffer = ReplayBuffer(N, 1, 1, num_steps, min_prio, 1, 1.0, 0.0, gamma, 0.4, 0.5)
    replay_buffer.priority_tree.update(
        torch.tensor([0, 2, 4], dtype=torch.int64), torch.full((N // 2,), 1e-2)
    )
    replay_buffer.priority_tree.update(
        torch.tensor([1, 3, 5], dtype=torch.int64), torch.full((N // 2,), 1e-1)
    )
    replay_buffer.terminals[:] = 0

    data_indices = torch.tensor([1, 3, 5], dtype=torch.int64)
    priorities = torch.tensor([0.0, 2e-2, 4e-2])

    replay_buffer.update_priorities(data_indices, priorities)
    assert replay_buffer.max_priority == pytest.approx(5e-2)

    print(replay_buffer.priority_tree.nodes[torch.arange(N, dtype=torch.int64) + N - 1])
    print(torch.tensor([1e-2, 4e-2, 1.5e-2, 4e-2, 2.5e-2, 5e-2]))

    assert_close(
        replay_buffer.priority_tree.nodes[torch.arange(N, dtype=torch.int64) + N - 1],
        torch.tensor([1e-2, 4e-2, 1.5e-2, 4e-2, 2.5e-2, 5e-2]),
    )
