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
        self.train_parser = self.subparsers.add_parser('train', help="Trains agent.")
        self.test_parser = self.subparsers.add_parser('test', help="Tests agent locally.")
        self.play_parser = self.subparsers.add_parser('play', help="Play online.")
        # Link commands
        self.train_parser.set_defaults(func=train)
        self.test_parser.set_defaults(func=test)
        self.play_parser.set_defaults(func=play)

        ## Global arguments
        self.main_parser.add_argument(
            '-v',
            '--verbose',
            default=False,
            action='store_true',
            help="Activates verbose console output.",
        )
        self.main_parser.add_argument(
            '--state-dim',
            type=int,
            default=18,
            help=".",
        )
        self.main_parser.add_argument(
            '--hidden-dim',
            type=int,
            default=512,
            help=".",
        )
        self.main_parser.add_argument(
            '--action-dim',
            type=int,
            default=4,
            help=".",
        )
        self.main_parser.add_argument(
            '--discretization-dim',
            type=int,
            default=3,
            help=".",
        )
        self.main_parser.add_argument(
            '--mode',
            type=int,
            default=0,
            help=".",
        )
        self.main_parser.add_argument(
            '--max-abs-force',
            type=float,
            default=1.0,
            help=".",
        )
        self.main_parser.add_argument(
            '--max-abs-torque',
            type=float,
            default=1.0,
            help=".",
        )
        self.main_parser.add_argument(
            '--no-gpu',
            default=False,
            action='store_true',
            help=".",
        )
        self.main_parser.add_argument(
            '--rng-seed',
            type=int,
            default=7,
            help=".",
        )
        self.main_parser.add_argument(
            '--model-filepath',
            type=str,
            default='model.pt',
            help=".",
        )
        self.main_parser.add_argument(
            '--num-eval-episodes',
            type=int,
            default=100,
            help=".",
        )
        self.main_parser.add_argument(
            '--disable-rendering',
            default=False,
            action='store_true',
            help=".",
        )
        self.main_parser.add_argument(
            '--disable-progress-bar',
            default=False,
            action='store_true',
            help=".",
        )

        ## Train arguments
        self.train_parser.add_argument(
            '--batch-size',
            type=int,
            default=256,
            help=".",
        )
        self.train_parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-4,
            help=".",
        )
        self.train_parser.add_argument(
            '--grad-clip-norm',
            type=float,
            default=40.0,
            help=".",
        )
        self.train_parser.add_argument(
            '--replay-buffer-size',
            type=int,
            default=1_000_000,
            help=".",
        )
        self.train_parser.add_argument(
            '--num-steps',
            type=int,
            default=3,
            help=".",
        )
        self.train_parser.add_argument(
            '--min-priority',
            type=float,
            default=1e-2,
            help=".",
        )
        self.train_parser.add_argument(
            '--alpha',
            type=float,
            default=0.6,
            help=".",
        )
        self.train_parser.add_argument(
            '--beta',
            type=float,
            default=0.2,
            help=".",
        )
        self.train_parser.add_argument(
            '--gamma',
            type=float,
            default=0.99,
            help=".",
        )
        self.train_parser.add_argument(
            '--epsilon-start',
            type=float,
            default=0.1,
            help=".",
        )
        self.train_parser.add_argument(
            '--epsilon-min',
            type=float,
            default=1e-4,
            help=".",
        )
        self.train_parser.add_argument(
            '--decay-factor',
            type=float,
            default=0.999999,
            help=".",
        )
        self.train_parser.add_argument(
            '--num-frames',
            type=int,
            default=10_000_000,
            help=".",
        )
        self.train_parser.add_argument(
            '--learn-freq',
            type=int,
            default=1,
            help=".",
        )
        self.train_parser.add_argument(
            '--update-target-freq',
            type=int,
            default=100,
            help=".",
        )
        self.train_parser.add_argument(
            '--num-warmup-frames',
            type=int,
            default=50_000,
            help=".",
        )
        self.train_parser.add_argument(
            '--continue-learning',
            default=False,
            action='store_true',
            help=".",
        )
        self.train_parser.add_argument(
            '--log-freq',
            type=int,
            default=100,
            help=".",
        )
        self.train_parser.add_argument(
            '--eval-freq',
            type=int,
            default=100_000,
            help=".",
        )

        ## Play arguments
        # TODO: Add arguments

    def __call__(self) -> None:
        args = self.main_parser.parse_args()
        args.func(**vars(args))
