import random
from collections import deque
from copy import deepcopy
from enum import IntEnum
from typing import Deque, Tuple

import laserhockey.hockey_env as h_env
import numpy as np
import torch
from tqdm import trange

from rl.agent import Agent


class Mode(IntEnum):
    """Environment modes."""

    TRAIN_DEF = 0
    TRAIN_ATK = 1
    PLAY_WEAK = 2
    PLAY_STRONG = 3
    PLAY_SELF = 4
    PLAY_WEAK_STRONG = 5
    PLAY_WEAK_STRONG_SELF = 6


def setup_rng(rng_seed: int) -> None:
    """
    Sets random number generators up by specifying the initial seed.

    Args:
        rng_seed (int): Random number generator seed. Set to negative values to generate a random seed.
    """

    if rng_seed >= 0:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
    else:
        rng_seed = torch.seed()
        random.seed(rng_seed)
        np.random.seed(rng_seed)


def get_device(no_gpu: bool) -> torch.device:
    """
    Returns device used for computations (CPU or GPU).

    Args:
        no_gpu (bool): Disables CUDA.

    Returns:
        torch.device: Device used for computations.
    """

    if no_gpu or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda")


def get_env_parameters() -> Tuple[int, int, float, float, float, float, float, float, float]:
    """
    Returns constant environment parameters.

    Returns:
        Tuple[int, int, float, float, float, float, float, float, float]: State dim, action dim, half field width, half field height, maximum absolute player velocity, maximum absolute player angle, maximum absolute player angular velocity, maximum absolute puck velocity, maximum time puck can be kept.
    """

    state_dim = 18
    action_dim = 4
    w = h_env.CENTER_X
    h = h_env.CENTER_Y
    vel = 10  # cf. L. 612
    ang = h_env.MAX_ANGLE
    ang_vel = 6 * np.pi
    vel_puck = 2 * h_env.MAX_PUCK_SPEED
    t = h_env.MAX_TIME_KEEP_PUCK

    return state_dim, action_dim, w, h, vel, ang, ang_vel, vel_puck, t


def get_env_from_mode(mode: int) -> h_env.HockeyEnv:
    """
    Returns hockey environment, specified by mode.

    Args:
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (self-play), 5 (play vs. weak and strong bot), 6 (play vs. weak and strong bot + self-play).

    Returns:
        h_env.HockeyEnv: Hockey environment.
    """

    if mode == Mode.TRAIN_DEF:
        return h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif mode == Mode.TRAIN_ATK:
        return h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    else:
        return h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)


def get_opponents_from_mode(
    agent: Agent,
    mode: int,
    max_num_opponents: int,
    num_episodes_weak: int,
    num_episodes_strong: int,
    num_episodes_self: int,
) -> Deque[Tuple[str, Agent | h_env.BasicOpponent, int]]:
    """
    Returns opponent agent, specified by mode.

    Args:
        agent (Agent): Protagonist RL agent.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (self-play), 5 (play vs. weak and strong bot), 6 (play vs. weak and strong bot + self-play).
        max_num_opponents (int): Maximum number of opponents.
        num_episodes_weak (int): Number of episodes agains weak bot until change.
        num_episodes_strong (int): Number of episodes agains strong bot until change.
        num_episodes_self (int): Number of episodes in self-play until change.

    Raises:
        ValueError: Modes above 6 are invalid.

    Returns:
        Deque[Tuple[str, Agent | h_env.BasicOpponent, int]]: Opponent name, Opponent agent, number of episodes until change.
    """

    opponents = deque(maxlen=max_num_opponents)

    if mode <= Mode.PLAY_WEAK:
        opponents.append(("Weak", h_env.BasicOpponent(weak=True), num_episodes_weak))
    elif mode == Mode.PLAY_STRONG:
        opponents.append(("Strong", h_env.BasicOpponent(weak=False), num_episodes_strong))
    elif mode == Mode.PLAY_SELF:
        opponents.append(("Self", deepcopy(agent), num_episodes_self))
    elif mode == Mode.PLAY_WEAK_STRONG:
        opponents.append(("Weak", h_env.BasicOpponent(weak=True), num_episodes_weak))
        opponents.append(("Strong", h_env.BasicOpponent(weak=False), num_episodes_strong))
    elif mode == Mode.PLAY_WEAK_STRONG_SELF:
        opponents.append(("Weak", h_env.BasicOpponent(weak=True), num_episodes_weak))
        opponents.append(("Strong", h_env.BasicOpponent(weak=False), num_episodes_strong))
        opponents.append(("Self", deepcopy(agent), num_episodes_self))
    else:
        raise ValueError(f"Invalid mode: {mode}.")

    return opponents


def update_opponents(
    agent: Agent,
    mode: int,
    opponents: Deque[Tuple[str, Agent | h_env.BasicOpponent, int]],
    num_episodes_self: int,
) -> None:
    """
    Updates self-play opponents in-place.

    Args:
        agent (Agent): Current learning agent.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (self-play), 5 (play vs. weak and strong bot), 6 (play vs. weak and strong bot + self-play).
        opponents (Deque[Tuple[str, Agent  |  h_env.BasicOpponent, int]]): Opponent name, Opponent agent, number of episodes until change.
        num_episodes_self (int): Number of episodes in self-play until change.
    """

    if mode == Mode.PLAY_SELF:
        # can simply add new self-play agent to FIFO
        opponents.append(("Self", deepcopy(agent), num_episodes_self))
    elif mode == Mode.PLAY_WEAK_STRONG_SELF:
        # save original FIFO length
        num_opponents = len(opponents)
        # first two items are bots, which should be preserved
        weak = opponents.popleft()
        strong = opponents.popleft()
        # remove old self-play agent, if FIFO was full before update start
        if num_opponents == opponents.maxlen:
            _ = opponents.popleft()
        # add weak and strong bots again
        opponents.appendleft(strong)
        opponents.appendleft(weak)
        # add new self-play agent
        opponents.append(("Self", deepcopy(agent), num_episodes_self))
    else:
        return


def play_eval(
    env: h_env.HockeyEnv,
    agent_p1: Agent,
    agent_p2: Agent | h_env.BasicOpponent,
    mode: int,
    num_episodes: int,
    disable_rendering: bool,
    disable_progress_bar: bool,
) -> Tuple[int, int, int]:
    """
    Starts evaluation of RL agent.

    Args:
        env (h_env.HockeyEnv): Hockey environment.
        agent_p1 (Agent): Protagonist RL agent.
        agent_p2 (Agent | h_env.BasicOpponent): Opponent agent.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (self-play), 5 (play vs. weak and strong bot), 6 (play vs. weak and strong bot + self-play).
        num_episodes (int): Number of evaluation episodes.
        disable_rendering (bool): Disables graphical rendering.
        disable_progress_bar (bool): Disables progress bar.

    Returns:
        Tuple[int, int, int]: Number of wins, draws, and defeats.
    """

    win_stats = np.zeros((num_episodes,))

    for episode_idx in trange(num_episodes, disable=disable_progress_bar):
        env.seed(np.random.randint(0, 100_000_000))
        state_p1, _ = env.reset()
        state_p2 = env.obs_agent_two()
        terminal = False

        while not terminal:
            if not disable_rendering:
                env.render()

            action_c_p1, _ = agent_p1.act(state_p1, eval_=True)
            if isinstance(agent_p2, Agent):
                action_c_p2, _ = agent_p2.act(state_p2, eval_=True)  # type: ignore
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


def compute_winning_percentage(num_wins: int, num_draws: int, num_games: int) -> float:
    return (num_wins + 0.5 * num_draws) / num_games
