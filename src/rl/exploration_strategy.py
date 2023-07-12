import torch


class EpsilonGreedy:
    """Implements base epsilon-greedy exploration strategy without decay."""

    def __init__(self, epsilon: float, discretization_dim: int) -> None:
        self.epsilon = epsilon
        self.discretization_dim = discretization_dim

    def decay(self) -> None:
        pass

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        u = torch.rand(1).item()
        # select random action
        if u < self.epsilon:
            action = torch.randint_like(action, 0, self.discretization_dim)

        # decay
        self.decay()

        return action


class EpsilonGreedyExpDecay(EpsilonGreedy):
    """Implements epsilon-greedy exploration strategy with exponential decay."""

    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        decay_factor: float,
        discretization_dim: int,
    ) -> None:
        super().__init__(epsilon_start, discretization_dim)
        self.epsilon_min = epsilon_min
        self.decay_factor = decay_factor

    def decay(self) -> None:
        self.epsilon = max(self.epsilon * self.decay_factor, self.epsilon_min)


# TODO: Increase Epsilon on Plateau
