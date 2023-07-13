import torch


class EpsilonGreedy:
    """Implements base epsilon-greedy exploration strategy without decay."""

    def __init__(self, epsilon: float, discretization_dim: int) -> None:
        """
        Initializes base epsilon-greedy exploration strategy.

        Args:
            epsilon (float): Epsilon (probability of choosing a random action).
            discretization_dim (int): Dimensionality of the action discretization.
        """

        self.epsilon = epsilon
        self.discretization_dim = discretization_dim

    def decay(self) -> None:
        """
        Performs decay w.r.t. epsilon (not in base epsilon-greedy).
        """

        pass

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """
        Selects action according to epsilon-greedy exploration strategy.
        A: Action dimension.
        D: Action discretization dimension.

        Args:
            action (torch.Tensor): Greedy action [A, D].

        Returns:
            torch.Tensor: Selected action [A, D].
        """

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
        """
        Initializes epsilon-greedy exploration strategy with exponential decay.

        Args:
            epsilon_start (float): Initial epsilon (probability of choosing a random action).
            epsilon_min (float): Minimum epsilon in the course of decay.
            decay_factor (float): Decay factor for epsilon.
            discretization_dim (int): Dimensionality of the action discretization.
        """

        super().__init__(epsilon_start, discretization_dim)
        self.epsilon_min = epsilon_min
        self.decay_factor = decay_factor

    def decay(self) -> None:
        """
        Performs exponential decay w.r.t. epsilon.
        """

        self.epsilon = max(self.epsilon * self.decay_factor, self.epsilon_min)


# TODO: Increase Epsilon on Plateau
