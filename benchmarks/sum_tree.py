import os
import sys
from argparse import ArgumentParser
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from tueplots import bundles

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
plt.rcParams.update(bundles.beamer_moml())

from src.rl.replay_buffer import SumTree as VecSumTree


class Timer:
    """Time measurements."""

    def __init__(self) -> None:
        self._start = None
        self._end = None

    def start(self) -> None:
        self._end = None
        self._start = default_timer()

    def end(self) -> None:
        self._end = default_timer()

    def elapsed_time(self) -> float:
        if self._start is not None and self._end is not None:
            return self._end - self._start
        else:
            return 0.0


class TimerContext:
    """Context handler for time measurements."""

    def __init__(self) -> None:
        self.timer = Timer()

    def __enter__(self) -> Timer:
        self.timer.start()
        return self.timer

    def __exit__(self, *args) -> None:
        self.timer.end()


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


def main():
    parser = ArgumentParser(
        "SumTree Benchmark",
        description="Benchmark script to compare basic sum tree implementation with the vectorized version of AlphaPuck.",
    )
    parser.add_argument("--min-treesize", type=int, default=1, help=".")
    parser.add_argument("--max-treesize", type=int, default=1_000_000, help=".")
    parser.add_argument("--num-treesize", type=int, default=7, help=".")
    parser.add_argument("--fixed-treesize", type=int, default=1_000_000, help=".")

    parser.add_argument("--min-samplesize", type=int, default=1, help=".")
    parser.add_argument("--max-samplesize", type=int, default=4096, help=".")
    parser.add_argument("--num-samplesize", type=int, default=13, help=".")
    parser.add_argument("--fixed-samplesize", type=int, default=256, help=".")

    parser.add_argument("--num-repetitions", type=int, default=10, help=".")

    args = parser.parse_args()

    min_base, max_base = np.log10(args.min_treesize), np.log10(args.max_treesize)
    tree_sizes = np.logspace(min_base, max_base, args.num_treesize, base=10.0, dtype=int)

    min_base, max_base = np.log2(args.min_samplesize), np.log2(args.max_samplesize)
    sample_sizes = np.logspace(min_base, max_base, args.num_samplesize, base=2.0, dtype=int)

    get_times_samplesize = np.zeros((2, args.num_samplesize, args.num_repetitions))
    update_times_samplesize = np.zeros_like(get_times_samplesize)
    get_times_treesize = np.zeros((2, args.num_treesize, args.num_repetitions))
    update_times_treesize = np.zeros_like(get_times_treesize)

    transform_uniform = (
        lambda uniform_sample, lower, upper: (lower - upper) * uniform_sample + upper
    )

    with tqdm(total=args.num_treesize * args.num_repetitions) as pbar:
        for idx1, tree_size in enumerate(tree_sizes):
            vec_sum_tree = VecSumTree(tree_size, 1, torch.int64)
            base_sum_tree = BaseSumTree(tree_size)

            data_indices = torch.multinomial(
                torch.full((tree_size,), 1 / tree_size),
                tree_size,
                replacement=True,
            ).to(torch.int64)
            data_indices = torch.sort(data_indices)[0]
            values = torch.rand((tree_size,), dtype=torch.float32)[data_indices]

            vec_sum_tree.update(data_indices, values)
            for idx3 in range(tree_size):
                base_sum_tree.update(data_indices[idx3].item(), values[idx3].item())

            for idx2 in range(args.num_repetitions):
                sample_size = min(tree_size, args.fixed_samplesize)

                data_indices = torch.multinomial(
                    torch.full((sample_size,), 1 / sample_size), sample_size, replacement=True
                ).to(torch.int64)
                data_indices = torch.sort(data_indices)[0]
                values = torch.rand((sample_size,), dtype=torch.float32)[data_indices]

                with TimerContext() as tc:
                    vec_sum_tree.update(data_indices, values)
                update_times_treesize[0, idx1, idx2] = tc.elapsed_time()

                with TimerContext() as tc:
                    for idx3 in range(sample_size):
                        base_sum_tree.update(data_indices[idx3].item(), values[idx3].item())
                update_times_treesize[1, idx1, idx2] = tc.elapsed_time()

                segment = vec_sum_tree.total / sample_size
                lower = segment * torch.arange(sample_size)
                upper = segment * torch.arange(1, sample_size + 1)
                cumsum = transform_uniform(
                    torch.rand(sample_size, dtype=torch.float32), lower, upper
                )

                with TimerContext() as tc:
                    vec_sum_tree.get(cumsum)
                get_times_treesize[0, idx1, idx2] = tc.elapsed_time()

                with TimerContext() as tc:
                    for idx3 in range(sample_size):
                        base_sum_tree.get(cumsum[idx3].item())
                get_times_treesize[1, idx1, idx2] = tc.elapsed_time()

                pbar.update()

    np.save("get_times_treesize.npy", get_times_treesize)
    np.save("update_times_treesize.npy", update_times_treesize)

    vec_sum_tree = VecSumTree(args.fixed_treesize, 1, torch.int64)
    base_sum_tree = BaseSumTree(args.fixed_treesize)

    data_indices = torch.multinomial(
        torch.full((args.fixed_treesize,), 1 / args.fixed_treesize),
        args.fixed_treesize,
        replacement=True,
    ).to(torch.int64)
    data_indices = torch.sort(data_indices)[0]
    values = torch.rand((args.fixed_treesize,), dtype=torch.float32)[data_indices]

    vec_sum_tree.update(data_indices, values)
    for idx3 in range(args.fixed_treesize):
        base_sum_tree.update(data_indices[idx3].item(), values[idx3].item())

    with tqdm(total=args.num_samplesize * args.num_repetitions) as pbar:
        for idx1, sample_size in enumerate(sample_sizes):
            for idx2 in range(args.num_repetitions):
                data_indices = torch.multinomial(
                    torch.full((sample_size,), 1 / sample_size), sample_size, replacement=True
                ).to(torch.int64)
                data_indices = torch.sort(data_indices)[0]
                values = torch.rand((sample_size,), dtype=torch.float32)[data_indices]

                with TimerContext() as tc:
                    vec_sum_tree.update(data_indices, values)
                update_times_samplesize[0, idx1, idx2] = tc.elapsed_time()

                with TimerContext() as tc:
                    for idx3 in range(sample_size):
                        base_sum_tree.update(data_indices[idx3].item(), values[idx3].item())
                update_times_samplesize[1, idx1, idx2] = tc.elapsed_time()

                segment = vec_sum_tree.total / sample_size
                lower = segment * torch.arange(sample_size)
                upper = segment * torch.arange(1, sample_size + 1)
                cumsum = transform_uniform(
                    torch.rand(sample_size, dtype=torch.float32), lower, upper
                )

                with TimerContext() as tc:
                    vec_sum_tree.get(cumsum)
                get_times_samplesize[0, idx1, idx2] = tc.elapsed_time()

                with TimerContext() as tc:
                    for idx3 in range(sample_size):
                        base_sum_tree.get(cumsum[idx3].item())
                get_times_samplesize[1, idx1, idx2] = tc.elapsed_time()

                pbar.update()

    np.save("get_times_samplesize.npy", get_times_samplesize)
    np.save("update_times_samplesize.npy", update_times_samplesize)

    # get_times_samplesize = np.load("get_times_samplesize.npy")
    # get_times_treesize = np.load("get_times_treesize.npy")
    # update_times_treesize = np.load("update_times_treesize.npy")

    plt.errorbar(
        sample_sizes,
        np.mean(get_times_samplesize[0], axis=-1),
        np.std(get_times_samplesize[0], axis=-1, ddof=1),
        label="Vectorized",
    )
    plt.errorbar(
        sample_sizes,
        np.mean(get_times_samplesize[1], axis=-1),
        np.std(get_times_samplesize[1], axis=-1, ddof=1),
        label="Iterative",
    )
    plt.title(
        f"SumTree Benchmark: Get operation over sample sizes at fixed tree size {args.fixed_treesize}"
    )
    plt.xlabel("Sample size")
    plt.xscale("log", base=2)
    plt.ylabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("get_times_samplesize.pdf")
    plt.clf()

    plt.errorbar(
        tree_sizes,
        np.mean(get_times_treesize[0], axis=-1),
        np.std(get_times_treesize[0], axis=-1, ddof=1),
        label="Vectorized",
    )
    plt.errorbar(
        tree_sizes,
        np.mean(get_times_treesize[1], axis=-1),
        np.std(get_times_treesize[1], axis=-1, ddof=1),
        label="Iterative",
    )
    plt.title(
        f"SumTree Benchmark: Get operation over tree sizes at fixed sample size {args.fixed_samplesize}"
    )
    plt.xlabel("Tree size")
    plt.xscale("log", base=10)
    plt.ylabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("get_times_treesize.pdf")
    plt.clf()

    plt.errorbar(
        sample_sizes,
        np.mean(update_times_samplesize[0], axis=-1),
        np.std(update_times_samplesize[0], axis=-1, ddof=1),
        label="Vectorized",
    )
    plt.errorbar(
        sample_sizes,
        np.mean(update_times_samplesize[1], axis=-1),
        np.std(update_times_samplesize[1], axis=-1, ddof=1),
        label="Iterative",
    )
    plt.title(
        f"SumTree Benchmark: Update operation over sample sizes at fixed tree size {args.fixed_treesize}"
    )
    plt.xlabel("Sample size")
    plt.xscale("log", base=2)
    plt.ylabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("update_times_samplesize.pdf")
    plt.clf()

    plt.errorbar(
        tree_sizes,
        np.mean(update_times_treesize[0], axis=-1),
        np.std(update_times_treesize[0], axis=-1, ddof=1),
        label="Vectorized",
    )
    plt.errorbar(
        tree_sizes,
        np.mean(update_times_treesize[1], axis=-1),
        np.std(update_times_treesize[1], axis=-1, ddof=1),
        label="Iterative",
    )
    plt.title(
        f"SumTree Benchmark: Update operation over tree sizes at fixed sample size {args.fixed_samplesize}"
    )
    plt.xlabel("Tree size")
    plt.xscale("log", base=10)
    plt.ylabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("update_times_treesize.pdf")
    plt.clf()


if __name__ == "__main__":
    main()
