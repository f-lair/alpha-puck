from collections import deque
from copy import deepcopy
from typing import Tuple

import laserhockey.hockey_env as h_env
import numpy as np

from rl.agent import Agent


class League:
    """Implements collection of opponents for 2-staged League Training."""

    def __init__(
        self, agent: Agent, size: int, weak_p: float, strong_p: float, self_p: float
    ) -> None:
        """
        Initializes league of opponents.

        Args:
            agent (Agent): Protagonist RL agent.
            size (int): Size of past-agents' league.
            weak_p (float): Probability of sampling weak opponent.
            strong_p (float): Probability of sampling strong opponent.
            self_p (float): Probability of sampling current agent as opponent.
        """

        assert 0 <= weak_p <= 1
        assert 0 <= strong_p <= 1
        assert 0 <= self_p <= 1

        past_self_p = 1.0 - weak_p - strong_p - self_p
        assert 0 <= past_self_p <= 1

        self.league1_p = np.array([weak_p, strong_p, self_p, past_self_p])
        self.league2_w = np.zeros((size,), dtype=int)
        self.league2_t = np.zeros((size,), dtype=int)

        self_agent = agent
        weak_agent = h_env.BasicOpponent(weak=True)
        strong_agent = h_env.BasicOpponent(weak=False)

        self.league1_agents = [weak_agent, strong_agent, self_agent]
        self.league2_agents = deque([deepcopy(self_agent)], maxlen=size)

    def sample(self) -> Tuple[Agent | h_env.BasicOpponent, int | None]:
        """
        Samples opponent from league.

        Returns:
            Tuple[Agent | h_env.BasicOpponent, int | None]: Sampled opponent.
        """

        league1_idx = np.random.choice(len(self.league1_p), p=self.league1_p)

        if league1_idx < len(self.league1_agents):
            return self.league1_agents[league1_idx], None
        else:
            league2_p = (
                1.0
                - np.divide(
                    self.league2_w,
                    2 * self.league2_t,
                    out=np.zeros(self.league2_w.shape, dtype=np.float64),
                    where=self.league2_t != 0,
                )
            ) ** 2
            league2_p /= league2_p.sum()

            league2_idx = np.random.choice(
                len(self.league2_agents), p=league2_p[: len(self.league2_agents)]
            )
            return self.league2_agents[league2_idx], league2_idx

    def update_statistics(self, num_points: int, num_games: int, league2_idx: int) -> None:
        """
        Updates statistics of past-agent in league.

        Args:
            num_points (int): Won points (Win: 2, Draw: 1, Defeat: 0).
            num_games (int): Number of played games.
            league2_idx (int): Index of past-agent in league.
        """

        self.league2_w[league2_idx] += num_points
        self.league2_t[league2_idx] += num_games

    def add_past_agent(self) -> None:
        """
        Adds new past-agent based on current agent.
        """

        self_agent = self.league1_agents[-1]
        full = len(self.league2_agents) == self.league2_agents.maxlen
        self.league2_agents.append(deepcopy(self_agent))
        if full:
            self.league2_w[0:-1] = self.league2_w[1:]
            self.league2_w[-1] = 0

            self.league2_t[0:-1] = self.league2_t[1:]
            self.league2_t[-1] = 0
