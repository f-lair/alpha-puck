from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from cli.utils import (
    Mode,
    compute_winning_percentage,
    get_device,
    get_env_from_mode,
    get_opponent_from_mode,
    play_eval,
    setup_rng,
)
from rl.agent import Agent
from rl.critic import Critic, CriticDQN
from rl.exploration_strategy import EpsilonGreedyExpDecay
from rl.replay_buffer import ReplayBuffer


def train(
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    discretization_dim: int,
    batch_size: int,
    learning_rate: float,
    grad_clip_norm: float,
    replay_buffer_size: int,
    num_steps: int,
    min_priority: float,
    alpha: float,
    beta: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    decay_factor: float,
    num_frames: int,
    mode: int,
    max_abs_force: float,
    max_abs_torque: float,
    learn_freq: int,
    update_target_freq: int,
    num_warmup_frames: int,
    no_gpu: bool,
    rng_seed: int,
    logging_dir: str,
    checkpoint: str,
    log_freq: int,
    eval_freq: int,
    num_eval_episodes: int,
    disable_rendering: bool,
    disable_progress_bar: bool,
    **kwargs,
) -> None:
    """
    CLI command for RL agent training.

    Args:
        state_dim (int): Dimensionality of the state space.
        hidden_dim (int): Dimensionality of the hidden layers in the critic model.
        action_dim (int): Dimensionality of the action space.
        discretization_dim (int): Dimensionality of the action discretization.
        batch_size (int): Batch size per learning step.
        learning_rate (float): Learning rate of the optimizer (Adam).
        grad_clip_norm (float): Maximum gradient norm above which gradients are clipped to.
        replay_buffer_size (int): Size of the replay buffer.
        num_steps (int): Number of steps in multi-step-return.
        min_priority (float): Minimum priority per transition in the replay buffer.
        alpha (float): Priority exponent in the replay buffer.
        beta (float): Importance sampling exponent in the replay buffer.
        gamma (float): Discount factor.
        epsilon_start (float): Initial value for epsilon in the epsilon-greedy exploration strategy.
        epsilon_min (float): Minimum value for epsilon in the epsilon-greedy exploration strategy.
        decay_factor (float): Decay factor for epsilon in the epsilon-greedy exploration strategy.
        num_frames (int): Total number of frames used for training.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (play vs. AI).
        max_abs_force (float): Maximum absolute force used for translation.
        max_abs_torque (float): Maximum absolute torque used for rotation.
        learn_freq (int): _description_
        update_target_freq (int): _description_
        num_warmup_frames (int): _description_
        no_gpu (bool): Disables CUDA.
        rng_seed (int): Random number generator seed. Set to negative values to generate a random seed.
        logging_dir (str): Logging directory.
        checkpoint (str): Path to checkpoint for evaluation/further training.
        log_freq (int): Number of frames after which certain statistics (e.g., epsilon) are logged.
        eval_freq (int): Number of frames after which an evaluation interlude is started.
        num_eval_episodes (int): Number of evaluation episodes.
        disable_rendering (bool): Disables graphical rendering.
        disable_progress_bar (bool): Disables progress bar.
    """

    hparams = locals().copy()
    del hparams["kwargs"]

    setup_rng(rng_seed)
    device = get_device(no_gpu)

    env = get_env_from_mode(mode)

    q_model = CriticDQN(state_dim, hidden_dim, action_dim, discretization_dim)
    if checkpoint != "":
        q_model.load(checkpoint)
    q_model = q_model.to(device)

    replay_buffer = ReplayBuffer(
        replay_buffer_size, state_dim, action_dim, num_steps, min_priority, alpha, beta, gamma
    )
    exploration_strategy = EpsilonGreedyExpDecay(
        epsilon_start, epsilon_min, decay_factor, discretization_dim
    )
    optimizer = optim.Adam(params=q_model.main_parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss(reduction="sum").to(device)  # type: ignore
    agent_p1 = Agent(
        q_model, discretization_dim, max_abs_force, max_abs_torque, device, exploration_strategy
    )
    agent_p2 = get_opponent_from_mode(agent_p1, mode)

    log_path = Path(logging_dir).joinpath(datetime.now().strftime("%m_%d_%y__%H_%M"))
    model_filepath = log_path.joinpath("checkpoint.pt")
    logger = SummaryWriter(str(log_path))
    hparam_metrics_dict = {"hparam/winning-percentage": 0.0}

    frame_idx = 0
    pbar_stats = {"loss": 0.0}

    with tqdm(total=num_frames, disable=disable_progress_bar, postfix=pbar_stats) as pbar:
        while frame_idx < num_frames:
            env.seed(np.random.randint(0, 100_000_000))
            state_p1, _ = env.reset()
            state_p2 = env.obs_agent_two()
            terminal = False
            cumulative_reward = 0.0

            while not terminal:
                if not disable_rendering:
                    env.render()

                if frame_idx % log_freq == 0:
                    logger.add_scalar("Epsilon", exploration_strategy.epsilon, frame_idx)

                action_c_p1, action_d_p1 = agent_p1.act(state_p1)
                if mode == Mode.PLAY_RL:
                    action_c_p2, action_d_p2 = agent_p2.act(state_p2)
                else:
                    action_c_p2 = agent_p2.act(state_p2)

                next_state_p1, reward, terminal, _, info = env.step(
                    np.hstack([action_c_p1, action_c_p2])
                )
                next_state_p2 = env.obs_agent_two()

                frame_idx += 1
                pbar.update()
                cumulative_reward += reward

                replay_buffer.store(state_p1, next_state_p1, action_d_p1, reward, terminal)
                if mode == Mode.PLAY_RL:
                    replay_buffer.store(state_p2, next_state_p2, action_d_p2, -reward, terminal)  # type: ignore

                state_p1 = next_state_p1
                state_p2 = next_state_p2

                if frame_idx >= num_warmup_frames:
                    if frame_idx % learn_freq == 0:
                        loss = agent_p1.learn(
                            replay_buffer, loss_fn, optimizer, batch_size, grad_clip_norm
                        )
                        logger.add_scalar("Loss", loss, frame_idx)
                        pbar_stats["loss"] = loss
                        pbar.set_postfix(pbar_stats)

                    if frame_idx % eval_freq == 0:
                        num_wins, num_draws, num_defeats = play_eval(
                            env, agent_p1, agent_p2, mode, num_eval_episodes, True, True
                        )
                        winning_percentage = compute_winning_percentage(
                            num_wins, num_draws, num_eval_episodes
                        )
                        logger.add_scalar("Evaluation-Wins", num_wins, frame_idx)
                        logger.add_scalar("Evaluation-Draws", num_draws, frame_idx)
                        logger.add_scalar("Evaluation-Defeats", num_defeats, frame_idx)
                        logger.add_scalar(
                            "Evaluation-Winning-Percentage", winning_percentage, frame_idx
                        )
                        hparam_metrics_dict["hparam/winning-percentage"] = winning_percentage
                        agent_p1.save_model(str(model_filepath))

                    if frame_idx % update_target_freq == 0:
                        agent_p1.update_target_network()

                    if terminal:
                        logger.add_scalar("Learning-Game-Outcome", info["winner"], frame_idx)
                        logger.add_scalar(
                            "Learning-Cumulative-Reward", cumulative_reward, frame_idx
                        )
    logger.add_hparams(hparams, hparam_metrics_dict, run_name="hparams")


def test(
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    discretization_dim: int,
    mode: int,
    max_abs_force: float,
    max_abs_torque: float,
    no_gpu: bool,
    rng_seed: int,
    checkpoint: str,
    num_eval_episodes: int,
    disable_rendering: bool,
    disable_progress_bar: bool,
    **kwargs,
) -> None:
    """
    CLI command for RL agent testing/evaluation.

    Args:
        state_dim (int): Dimensionality of the state space.
        hidden_dim (int): Dimensionality of the hidden layers in the critic model.
        action_dim (int): Dimensionality of the action space.
        discretization_dim (int): Dimensionality of the action discretization.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (play vs. AI).
        max_abs_force (float): Maximum absolute force used for translation.
        max_abs_torque (float): Maximum absolute torque used for rotation.
        no_gpu (bool): Disables CUDA.
        rng_seed (int): Random number generator seed. Set to negative values to generate a random seed.
        checkpoint (str): Path to checkpoint for evaluation/further training.
        num_eval_episodes (int): Number of evaluation episodes.
        disable_rendering (bool): Disables graphical rendering.
        disable_progress_bar (bool): Disables progress bar.
    """

    setup_rng(rng_seed)

    device = get_device(no_gpu)

    env = get_env_from_mode(mode)

    q_model = CriticDQN(state_dim, hidden_dim, action_dim, discretization_dim)
    q_model.load(checkpoint)
    q_model = q_model.to(device)

    agent_p1 = Agent(q_model, discretization_dim, max_abs_force, max_abs_torque, device)
    agent_p2 = get_opponent_from_mode(agent_p1, mode)

    num_wins, num_draws, num_defeats = play_eval(
        env, agent_p1, agent_p2, mode, num_eval_episodes, disable_rendering, disable_progress_bar
    )
    winning_percentage = compute_winning_percentage(num_wins, num_draws, num_eval_episodes)

    print(f"Wins: {num_wins} / {num_eval_episodes} ({num_wins / num_eval_episodes:.1%})")
    print(f"Draws: {num_draws} / {num_eval_episodes} ({num_draws / num_eval_episodes:.1%})")
    print(f"Defeats: {num_defeats} / {num_eval_episodes} ({num_defeats / num_eval_episodes:.1%})")
    print(f"Winning percentage: {winning_percentage:.1%}")


def play(**kwargs) -> None:
    """
    CLI command for RL agent inference/live play.

    Raises:
        NotImplementedError: TBD.
    """

    # TODO: Implement local and remote play
    raise NotImplementedError
