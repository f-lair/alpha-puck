import random
from enum import IntEnum
from typing import Tuple

import laserhockey.hockey_env as h_env
import numpy as np
import torch
from tqdm import trange

from rl.agent import Agent


class Mode(IntEnum):
    TRAIN_DEF = 0
    TRAIN_ATK = 1
    PLAY_WEAK = 2
    PLAY_STRONG = 3
    PLAY_RL = 4


def setup_rng(rng_seed: int) -> None:
    if rng_seed >= 0:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
    else:
        rng_seed = torch.seed()
        random.seed(rng_seed)
        np.random.seed(rng_seed)


def get_device(no_gpu: bool) -> torch.device:
    if no_gpu or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda")


def get_env_from_mode(mode: int) -> h_env.HockeyEnv:
    if mode == Mode.TRAIN_DEF:
        return h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif mode == Mode.TRAIN_ATK:
        return h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    else:
        return h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)


def get_opponent_from_mode(agent: Agent, mode: int) -> Agent | h_env.BasicOpponent:
    if mode <= Mode.PLAY_WEAK:
        return h_env.BasicOpponent(weak=True)
    elif mode == Mode.PLAY_STRONG:
        return h_env.BasicOpponent(weak=False)
    elif mode == Mode.PLAY_RL:
        return agent
    else:
        raise ValueError(f"Invalid mode: {mode}.")


def play_eval(
    env: h_env.HockeyEnv,
    agent_p1: Agent,
    agent_p2: Agent | h_env.BasicOpponent,
    mode: int,
    num_episodes: int,
    disable_rendering: bool,
    disable_progress_bar: bool,
) -> Tuple[int, int, int]:
    win_stats = np.zeros((num_episodes,))

    for episode_idx in trange(num_episodes, disable=disable_progress_bar):
        env.seed(np.random.randint(0, 100_000_000))
        state_p1, _ = env.reset()
        state_p2 = env.obs_agent_two()
        terminal = False

        while not terminal:
            if not disable_rendering:
                env.render()

            action_c_p1, _ = agent_p1.act(state_p1)
            if mode == Mode.PLAY_RL:
                action_c_p2, _ = agent_p2.act(state_p2)
            else:
                action_c_p2 = agent_p2.act(state_p2)

            state_p1, _, terminal, _, info = env.step(np.hstack([action_c_p1, action_c_p2]))
            state_p2 = env.obs_agent_two()

            if terminal:
                win_stats[episode_idx] = info["winner"]

    num_wins = (win_stats == 1).sum()
    num_draws = (win_stats == 0).sum()
    num_defeats = (win_stats == -1).sum()

    return num_wins, num_draws, num_defeats
