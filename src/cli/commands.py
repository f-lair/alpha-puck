from datetime import datetime
from pathlib import Path

import numpy as np
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from cli.utils import (
    Mode,
    compute_winning_percentage,
    get_device,
    get_env_from_mode,
    get_env_parameters,
    get_league_from_mode,
    play_eval,
    setup_rng,
)
from remote.client.backend.client import Client
from rl.agent import Agent
from rl.critic import Critic, CriticDQN
from rl.exploration_strategy import EpsilonGreedyExpDecay
from rl.replay_buffer import ReplayBuffer


def train(
    agent_name: str,
    hidden_dim: int,
    discretization_dim: int,
    no_state_norm: bool,
    batch_size: int,
    learning_rate: float,
    grad_clip_norm: float,
    replay_buffer_size: int,
    num_steps: int,
    min_priority: float,
    decay_window: int,
    alpha: float,
    beta: float,
    gamma: float,
    nu: float,
    rho: float,
    epsilon_start: float,
    epsilon_min: float,
    decay_factor: float,
    num_frames: int,
    mode: int,
    league_size: int,
    weak_p: float,
    strong_p: float,
    self_p: float,
    league_change_freq: int,
    max_abs_force: float,
    max_abs_torque: float,
    learn_freq: int,
    update_target_freq: int,
    num_warmup_frames: int,
    no_gpu: bool,
    rng_seed: int,
    logging_dir: str,
    logging_name: str,
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
        agent_name (str): Agent name.
        hidden_dim (int): Dimensionality of the hidden layers in the critic model.
        discretization_dim (int): Dimensionality of the action discretization.
        no_state_norm (bool): Disables state normalization.
        batch_size (int): Batch size per learning step.
        learning_rate (float): Learning rate of the optimizer (Adam).
        grad_clip_norm (float): Maximum gradient norm above which gradients are clipped to.
        replay_buffer_size (int): Size of the replay buffer.
        num_steps (int): Number of steps in multi-step-return.
        min_priority (float): Minimum priority per transition in the replay buffer.
        decay_window (int): Size of the decay window in PSER. Set to 1 for regular PER behavior.
        alpha (float): Priority exponent in the replay buffer.
        beta (float): Importance sampling exponent in the replay buffer.
        gamma (float): Discount factor.
        nu (float): Previous priority in PSER.
        rho (float): Decay coefficient in PSER.
        epsilon_start (float): Initial value for epsilon in the epsilon-greedy exploration strategy.
        epsilon_min (float): Minimum value for epsilon in the epsilon-greedy exploration strategy.
        decay_factor (float): Decay factor for epsilon in the epsilon-greedy exploration strategy.
        num_frames (int): Total number of frames used for training.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (self-play), 5 (play vs. weak and strong bot), 6 (play vs. weak and strong bot + self-play).
        league_size (int): Size of past-agents' league.
        weak_p (float): Probability of sampling weak opponent.
        strong_p (float): Probability of sampling strong opponent.
        self_p (float): Probability of sampling current agent as opponent.
        league_change_freq (int): Number of frames after which past-agents are updated.
        max_abs_force (float): Maximum absolute force used for translation.
        max_abs_torque (float): Maximum absolute torque used for rotation.
        learn_freq (int): Number of frames after which a learning step is performed.
        update_target_freq (int): Number of frames after which the target critic is updated.
        num_warmup_frames (int): Number of initial frames before learning is started.
        no_gpu (bool): Disables CUDA.
        rng_seed (int): Random number generator seed. Set to negative values to generate a random seed.
        logging_dir (str): Logging directory.
        logging_name (str): Logging run name. Defaults to date and time, if empty.
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

    state_dim, action_dim, w, h, vel, ang, ang_vel, vel_puck, t = get_env_parameters()

    env = get_env_from_mode(mode)

    q_model = CriticDQN(
        state_dim,
        hidden_dim,
        action_dim,
        discretization_dim,
        no_state_norm,
        w,
        h,
        vel,
        ang,
        ang_vel,
        vel_puck,
        t,
        device,
    )
    if checkpoint != "":
        q_model.load(checkpoint)
    q_model = q_model.to(device)

    replay_buffer = ReplayBuffer(
        replay_buffer_size,
        state_dim,
        action_dim,
        num_steps,
        min_priority,
        decay_window,
        alpha,
        beta,
        gamma,
        nu,
        rho,
        device,
    )
    exploration_strategy = EpsilonGreedyExpDecay(
        epsilon_start, epsilon_min, decay_factor, discretization_dim
    )
    optimizer = optim.Adam(params=q_model.main_parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss(reduction="sum").to(device)  # type: ignore
    agent_p1 = Agent(
        agent_name,
        q_model,
        discretization_dim,
        max_abs_force,
        max_abs_torque,
        device,
        exploration_strategy,
    )
    league = get_league_from_mode(
        agent_p1,
        mode,
        league_size,
        weak_p,
        strong_p,
        self_p,
    )
    eval_opponents = {"Weak": league.league1_agents[0], "Strong": league.league1_agents[1]}

    if logging_name == "":
        logging_name = datetime.now().strftime("%m_%d_%y__%H_%M")
    log_path = Path(logging_dir).joinpath(logging_name)
    logger = SummaryWriter(str(log_path))
    hparam_metrics_dict = {
        "hparam/weak-winning-percentage": 0.0,
        "hparam/strong-winning-percentage": 0.0,
    }

    frame_idx = 0
    log_terminal_idx = 0
    league_terminal_idx = 0
    pbar_stats = {"loss": 0.0}

    with tqdm(total=num_frames, disable=disable_progress_bar, postfix=pbar_stats) as pbar:
        while frame_idx < num_frames:
            env.seed(np.random.randint(0, 100_000_000))
            state_p1, _ = env.reset()
            state_p2 = env.obs_agent_two()
            terminal = False
            cumulative_reward = 0.0

            agent_p2, league2_idx = league.sample()

            while not terminal:
                if not disable_rendering:
                    env.render()

                if frame_idx % log_freq == 0:
                    logger.add_scalar("Epsilon", exploration_strategy.epsilon, frame_idx)

                action_c_p1, action_d_p1 = agent_p1.act(state_p1)
                if isinstance(agent_p2, Agent):
                    action_c_p2, _ = agent_p2.act(state_p2, eval_=True)
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
                        for agent_p2_name, agent_p2 in eval_opponents.items():
                            num_wins, num_draws, num_defeats = play_eval(
                                env, agent_p1, agent_p2, num_eval_episodes, True, True
                            )
                            winning_percentage = compute_winning_percentage(
                                num_wins, num_draws, num_eval_episodes
                            )
                            logger.add_scalar(
                                f"Evaluation-{agent_p2_name}-Wins", num_wins, frame_idx
                            )
                            logger.add_scalar(
                                f"Evaluation-{agent_p2_name}-Draws", num_draws, frame_idx
                            )
                            logger.add_scalar(
                                f"Evaluation-{agent_p2_name}-Defeats", num_defeats, frame_idx
                            )
                            logger.add_scalar(
                                f"Evaluation-{agent_p2_name}-Winning-Percentage",
                                winning_percentage,
                                frame_idx,
                            )
                            hparam_metrics_dict[
                                f"hparam/{agent_p2_name.lower()}-winning-percentage"
                            ] = winning_percentage
                        model_filepath = log_path.joinpath(f"checkpoint-{frame_idx}.pt")
                        agent_p1.save_model(str(model_filepath))

                    if frame_idx % update_target_freq == 0:
                        agent_p1.update_target_network()

                    if terminal:
                        if league2_idx is not None:
                            league.update_statistics(info["winner"] + 1, 1, league2_idx)
                        if frame_idx // eval_freq >= log_terminal_idx:
                            logger.add_scalar("Learning-Game-Outcome", info["winner"], frame_idx)
                            logger.add_scalar(
                                "Learning-Cumulative-Reward", cumulative_reward, frame_idx
                            )
                            log_terminal_idx += 1
                        if frame_idx // league_change_freq > league_terminal_idx:
                            league.add_past_agent()
                            league_terminal_idx += 1
    logger.add_hparams(hparams, hparam_metrics_dict, run_name="hparams")

    logger.close()
    env.close()


def test(
    agent_name: str,
    hidden_dim: int,
    discretization_dim: int,
    no_state_norm: bool,
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
        agent_name (str): Agent name.
        hidden_dim (int): Dimensionality of the hidden layers in the critic model.
        discretization_dim (int): Dimensionality of the action discretization.
        no_state_norm (bool): Disables state normalization.
        mode (int): Environment mode: 0 (defense), 1 (attacking), 2 (play vs. weak bot), 3 (play vs. strong bot), 4 (self-play).
        max_abs_force (float): Maximum absolute force used for translation.
        max_abs_torque (float): Maximum absolute torque used for rotation.
        no_gpu (bool): Disables CUDA.
        rng_seed (int): Random number generator seed. Set to negative values to generate a random seed.
        checkpoint (str): Path to checkpoint for evaluation/further training.
        num_eval_episodes (int): Number of evaluation episodes.
        disable_rendering (bool): Disables graphical rendering.
        disable_progress_bar (bool): Disables progress bar.
    """

    assert mode <= Mode.PLAY_SELF, "Evaluation is supported only agains single opponents!"

    setup_rng(rng_seed)
    device = get_device(no_gpu)

    state_dim, action_dim, w, h, vel, ang, ang_vel, vel_puck, t = get_env_parameters()

    env = get_env_from_mode(mode)

    q_model = Critic(
        state_dim,
        hidden_dim,
        action_dim,
        discretization_dim,
        no_state_norm,
        w,
        h,
        vel,
        ang,
        ang_vel,
        vel_puck,
        t,
        device,
    )
    q_model.load(checkpoint)
    q_model = q_model.to(device)

    agent_p1 = Agent(
        agent_name, q_model, discretization_dim, max_abs_force, max_abs_torque, device
    )
    league = get_league_from_mode(agent_p1, mode, 1, 1.0, 1.0, 1.0)
    agent_p2, _ = league.sample()

    num_wins, num_draws, num_defeats = play_eval(
        env, agent_p1, agent_p2, num_eval_episodes, disable_rendering, disable_progress_bar
    )
    winning_percentage = compute_winning_percentage(num_wins, num_draws, num_eval_episodes)

    print(f"Wins: {num_wins} / {num_eval_episodes} ({num_wins / num_eval_episodes:.1%})")
    print(f"Draws: {num_draws} / {num_eval_episodes} ({num_draws / num_eval_episodes:.1%})")
    print(f"Defeats: {num_defeats} / {num_eval_episodes} ({num_defeats / num_eval_episodes:.1%})")
    print(f"Winning percentage: {winning_percentage:.1%}")

    env.close()


def play(
    agent_name: str,
    hidden_dim: int,
    discretization_dim: int,
    no_state_norm: bool,
    max_abs_force: float,
    max_abs_torque: float,
    no_gpu: bool,
    rng_seed: int,
    logging_dir: str,
    logging_name: str,
    checkpoint: str,
    num_eval_episodes: int,
    user_name: str,
    user_password: str,
    **kwargs,
) -> None:
    """
    CLI command for RL agent remote play.

    Args:
        agent_name (str): Agent name.
        hidden_dim (int): Dimensionality of the hidden layers in the critic model.
        discretization_dim (int): Dimensionality of the action discretization.
        no_state_norm (bool): Disables state normalization.
        max_abs_force (float): Maximum absolute force used for translation.
        max_abs_torque (float): Maximum absolute torque used for rotation.
        no_gpu (bool): Disables CUDA.
        rng_seed (int): Random number generator seed. Set to negative values to generate a random seed.
        logging_dir (str): Logging directory.
        logging_name (str): Logging run name. Defaults to date and time, if empty.
        checkpoint (str): Path to checkpoint for evaluation/further training.
        num_eval_episodes (int): Number of evaluation episodes. Unlimited, if negative.
        user_name (str): Remote user name.
        user_password (str): Remote user password.
    """

    setup_rng(rng_seed)
    device = get_device(no_gpu)

    state_dim, action_dim, w, h, vel, ang, ang_vel, vel_puck, t = get_env_parameters()

    q_model = Critic(
        state_dim,
        hidden_dim,
        action_dim,
        discretization_dim,
        no_state_norm,
        w,
        h,
        vel,
        ang,
        ang_vel,
        vel_puck,
        t,
        device,
    )
    q_model.load(checkpoint)
    q_model = q_model.to(device)

    agent_p1 = Agent(
        agent_name, q_model, discretization_dim, max_abs_force, max_abs_torque, device
    )

    if logging_name == "":
        logging_name = datetime.now().strftime("%m_%d_%y__%H_%M")
    log_path = Path(logging_dir).joinpath(logging_name)

    client = Client(
        username=user_name,
        password=user_password,
        controller=agent_p1,
        output_path=str(log_path),
        interactive=False,
        op='start_queuing',
        num_games=num_eval_episodes if num_eval_episodes >= 0 else None,
    )
