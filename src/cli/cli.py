from argparse import ArgumentParser

from cli.commands import play, test, train


class CLI:
    """Implements Command Line Interface (CLI)."""

    def __init__(self) -> None:
        """
        Defines and initializes CLI.
        """

        ## Parser definitions
        self.main_parser = ArgumentParser(
            "AlphaPuck | DecQN",
            description="Main script for training, local testing and online playing with the DecQN implementation of AlphaPuck.",
        )
        self.subparsers = self.main_parser.add_subparsers(title="Commands")
        self.train_parser = self.subparsers.add_parser("train", help="Trains agent.")
        self.test_parser = self.subparsers.add_parser("test", help="Tests agent locally.")
        self.play_parser = self.subparsers.add_parser("play", help="Play online.")
        # Link commands
        self.train_parser.set_defaults(func=train)
        self.test_parser.set_defaults(func=test)
        self.play_parser.set_defaults(func=play)

        ## Global arguments
        self.main_parser.add_argument(
            "-v",
            "--verbose",
            default=False,
            action="store_true",
            help="Activates verbose console output.",
        )
        self.main_parser.add_argument(
            "--hidden-dim",
            type=int,
            default=512,
            help="Dimensionality of the hidden layers in the critic model.",
        )
        self.main_parser.add_argument(
            "--discretization-dim",
            type=int,
            default=3,
            help="Dimensionality of the action discretization.",
        )
        self.main_parser.add_argument(
            "--no-state-norm",
            default=False,
            action="store_true",
            help="Disables state normalization.",
        )
        self.main_parser.add_argument(
            "--mode",
            type=int,
            default=0,
            help="Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (play vs. AI), 5 (play vs. weak and strong bot).",
        )
        self.main_parser.add_argument(
            "--change-opponent-freq",
            type=int,
            default=1000,
            help="Number of episodes after which opponents are changed in mode 4.",
        )
        self.main_parser.add_argument(
            "--max-abs-force",
            type=float,
            default=1.0,
            help="Maximum absolute force used for translation.",
        )
        self.main_parser.add_argument(
            "--max-abs-torque",
            type=float,
            default=1.0,
            help="Maximum absolute torque used for rotation.",
        )
        self.main_parser.add_argument(
            "--no-gpu",
            default=False,
            action="store_true",
            help="Disables CUDA.",
        )
        self.main_parser.add_argument(
            "--rng-seed",
            type=int,
            default=7,
            help="Random number generator seed. Set to negative values to generate a random seed.",
        )
        self.main_parser.add_argument(
            "--checkpoint",
            type=str,
            default="",
            help="Path to checkpoint for evaluation/further training.",
        )
        self.main_parser.add_argument(
            "--num-eval-episodes",
            type=int,
            default=100,
            help="Number of evaluation episodes.",
        )
        self.main_parser.add_argument(
            "--disable-rendering",
            default=False,
            action="store_true",
            help="Disables graphical rendering.",
        )
        self.main_parser.add_argument(
            "--disable-progress-bar",
            default=False,
            action="store_true",
            help="Disables progress bar.",
        )

        ## Train arguments
        self.train_parser.add_argument(
            "--batch-size",
            type=int,
            default=256,
            help="Batch size per learning step.",
        )
        self.train_parser.add_argument(
            "--learning-rate",
            type=float,
            default=1e-4,
            help="Learning rate of the optimizer (Adam).",
        )
        self.train_parser.add_argument(
            "--grad-clip-norm",
            type=float,
            default=40.0,
            help="Maximum gradient norm above which gradients are clipped to.",
        )
        self.train_parser.add_argument(
            "--replay-buffer-size",
            type=int,
            default=1_000_000,
            help="Size of the replay buffer.",
        )
        self.train_parser.add_argument(
            "--num-steps",
            type=int,
            default=3,
            help="Number of steps in multi-step-return.",
        )
        self.train_parser.add_argument(
            "--min-priority",
            type=float,
            default=1e-2,
            help="Minimum priority per transition in the replay buffer.",
        )
        self.train_parser.add_argument(
            "--decay-window",
            type=int,
            default=5,
            help="Size of the decay window in PSER. Set to 1 for regular PER behavior.",
        )
        self.train_parser.add_argument(
            "--alpha",
            type=float,
            default=0.6,
            help="Priority exponent in the replay buffer.",
        )
        self.train_parser.add_argument(
            "--beta",
            type=float,
            default=0.2,
            help="Importance sampling exponent in the replay buffer.",
        )
        self.train_parser.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            help="Discount factor.",
        )
        self.train_parser.add_argument(
            "--nu",
            type=float,
            default=0.7,
            help="Previous priority in PSER.",
        )
        self.train_parser.add_argument(
            "--rho",
            type=float,
            default=0.4,
            help="Decay coefficient in PSER.",
        )
        self.train_parser.add_argument(
            "--epsilon-start",
            type=float,
            default=0.1,
            help="Initial value for epsilon in the epsilon-greedy exploration strategy.",
        )
        self.train_parser.add_argument(
            "--epsilon-min",
            type=float,
            default=1e-4,
            help="Minimum value for epsilon in the epsilon-greedy exploration strategy.",
        )
        self.train_parser.add_argument(
            "--decay-factor",
            type=float,
            default=0.999999,
            help="Decay factor for epsilon in the epsilon-greedy exploration strategy.",
        )
        self.train_parser.add_argument(
            "--num-frames",
            type=int,
            default=10_000_000,
            help="Total number of frames used for training.",
        )
        self.train_parser.add_argument(
            "--learn-freq",
            type=int,
            default=1,
            help="Number of frames after which a learning step is performed.",
        )
        self.train_parser.add_argument(
            "--update-target-freq",
            type=int,
            default=100,
            help="Number of frames after which the target critic is updated.",
        )
        self.train_parser.add_argument(
            "--num-warmup-frames",
            type=int,
            default=300,
            help="Number of initial frames before learning is started.",
        )
        self.train_parser.add_argument(
            "--logging-dir",
            type=str,
            default="../runs/",
            help="Logging directory.",
        )
        self.train_parser.add_argument(
            "--logging-name",
            type=str,
            default="",
            help="Logging run name. Defaults to date and time, if empty.",
        )
        self.train_parser.add_argument(
            "--log-freq",
            type=int,
            default=10_000,
            help="Number of frames after which certain statistics (e.g., epsilon) are logged.",
        )
        self.train_parser.add_argument(
            "--eval-freq",
            type=int,
            default=100_000,
            help="Number of frames after which an evaluation interlude is started.",
        )

        ## Play arguments
        # TODO: Add arguments

    def __call__(self) -> None:
        args = self.main_parser.parse_args()
        args.func(**vars(args))
