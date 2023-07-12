import math

import pytest
import torch
from src.rl.replay_buffer import SumTree


class BaseSumTree:
    """
    Base SumTree implementation
    cf. https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py
    """

    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]


def check_summing(nodes):
    correct = True
    num_levels = int(math.log2(len(nodes)))
    for level_idx in range(num_levels):
        num_checks = 2**level_idx
        level_base_idx = 2**level_idx - 1
        next_level_base_idx = 2 ** (level_idx + 1) - 1
        for check_idx in range(num_checks):
            parent_idx = level_base_idx + check_idx
            child_1_idx = next_level_base_idx + 2 * check_idx
            child_2_idx = child_1_idx + 1
            if child_1_idx >= len(nodes):
                break
            correct &= nodes[parent_idx] == nodes[child_1_idx] + nodes[child_2_idx]

    return correct


N = 16


@pytest.fixture
def example_sum_tree():
    sum_tree = SumTree(N, 1, torch.int64)

    for idx in range(N):
        sum_tree.add(float(idx), torch.tensor([N - 1 - idx]))

    return sum_tree


@pytest.fixture
def example_base_sum_tree():
    base_sum_tree = BaseSumTree(N)

    for idx in range(N):
        base_sum_tree.add(float(idx), N - 1 - idx)

    return base_sum_tree


def test_compare_to_base(example_sum_tree, example_base_sum_tree):
    size = example_base_sum_tree.size
    for idx in range(size):
        assert example_base_sum_tree.data[idx] == example_sum_tree.data[idx]

    size = 2 * example_base_sum_tree.size - 1
    for idx in range(size):
        assert example_base_sum_tree.nodes[idx] == example_sum_tree.nodes[idx]


def test_add(example_sum_tree):
    assert check_summing(example_sum_tree.nodes)


def test_minimum(example_sum_tree):
    value, data, data_idx = example_sum_tree.minimum

    assert value == 0.0
    assert data == N - 1
    assert data_idx == 0


def test_update():
    def update_base_sum_tree(base_sum_tree, data_indices, values):
        for idx in range(len(data_indices)):
            base_sum_tree.update(data_indices[idx].item(), values[idx].item())
        return base_sum_tree

    def compare_to_base(sum_tree, base_sum_tree):
        size = 2 * base_sum_tree.size - 1

        correct = torch.ones((size,))

        for idx in range(size):
            correct[idx] = base_sum_tree.nodes[idx] == pytest.approx(sum_tree.nodes[idx].item())

        return correct

    sum_tree = SumTree(N, 1, torch.int64)
    data_indices = torch.multinomial(torch.full((N,), 1 / N), N, replacement=True).to(torch.int64)
    data_indices = torch.sort(data_indices)[0]
    values = N * torch.rand((N,), dtype=torch.float32)[data_indices]
    sum_tree.update(data_indices, values)

    base_sum_tree = BaseSumTree(N)
    base_sum_tree = update_base_sum_tree(base_sum_tree, data_indices, values)

    correct = compare_to_base(sum_tree, base_sum_tree)
    assert torch.all(correct)

    values = torch.flip(values, [0])
    sum_tree.update(data_indices, values)
    base_sum_tree = update_base_sum_tree(base_sum_tree, data_indices, values)

    compare_to_base(sum_tree, base_sum_tree)
    assert torch.all(correct)


def test_get(example_sum_tree, example_base_sum_tree):
    cumsum = torch.tensor([1.0, 7.0, 84.0, 42.0, 58.0])

    data_indices, values, data = example_sum_tree.get(cumsum.clone())

    for idx in range(len(cumsum)):
        datum_idx, value, datum = example_base_sum_tree.get(cumsum[idx].item())

        assert data_indices[idx] == datum_idx
        assert values[idx] == value
        assert data[idx] == datum
