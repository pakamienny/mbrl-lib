# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import gym
import numpy as np
import omegaconf
import torch
import hydra

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.diagnostics.eval_model_on_dataset import DatasetEvaluator
import pathlib
import copy

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

def evaluate(
    env: gym.Env,
    agent: mbrl.planning.Agent,
    number_episodes: int = 10,
    replay_buffer = None
):
    agent.reset()
    rewards = mbrl.util.common.rollout_agent_trajectories(
        env,
        agent=agent,
        agent_kwargs={},
        steps_or_trials_to_collect=number_episodes,
        collect_full_trajectories=True,
        replay_buffer=replay_buffer
    )


    assert len(rewards)==number_episodes, "problem with number of rewards"
    return replay_buffer, (np.mean(rewards), np.std(rewards)/np.sqrt(number_episodes))

def plot_states(replay_buffer, path):
    pathlib.Path.mkdir(path, parents=True, exist_ok=True)
    import seaborn as sns
    states = replay_buffer.get("obs")
    distplot = sns.displot(states, kind="kde")
    fig = distplot.fig
    fig.savefig(path / "out.png") 
    



def create_trainer(cfg, dynamics_model, logger):
    model_trainer =  hydra.utils.instantiate(cfg.algorithm.model_trainer, 
                                             model=dynamics_model,
                                             logger=logger)
    return model_trainer

def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)
    evaluation_env = copy.deepcopy(env)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)



    initalize_model = [[
                        {"regressor": "X1+X2", "regressor_variance": None, "scale_x": None, "scale_y": None},
                        {"regressor": "exp(abs(X1+X2)/3)*cos(2*3.14159265359*(X1+X2))", "regressor_variance": None, "scale_x": None, "scale_y": None}
                    ],
                    [
                        {"regressor": "X1+X2", "regressor_variance": None, "scale_x": None, "scale_y": None},
                        {"regressor": "exp(abs(X1+X2)/3)*cos(2*3.14159265359*(X1+X2))", "regressor_variance": None, "scale_x": None, "scale_y": None}
                    ]
                    ]
    initalize_model = None
    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    model_trainer = create_trainer(cfg, dynamics_model, logger)


    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    replay_buffer.save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )

    if  cfg.algorithm.agent == "mbrl.planning.RandomAgent":
        agent = mbrl.planning.RandomAgent(env)
    else:
        agent = mbrl.planning.create_trajectory_optim_agent_for_model(
            model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
        )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    while env_steps < cfg.overrides.num_steps:
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                    save_model_all_epochs=cfg.save_model_all_epochs,
                    initalize_model=initalize_model if env_steps == 0 else None
                )
           
                to_log = {"env_step": cfg.algorithm.initial_exploration_steps+env_steps}

                if cfg.evaluate.evaluate_model_accuracies:
                    model_path = pathlib.Path(work_dir)

                    results_path = model_path / "diagnostics" / "dataset"
                    evaluator = DatasetEvaluator(model_path, model_path, results_path / "in_domain" / "epoch_{}".format(model_trainer._train_iteration))
                    metrics = evaluator.run() 
                    for k, v in metrics.items():
                        to_log["in_domain_{}".format(k)]=v
   
                    datasets_paths = cfg.overrides.evaluate_dataset_path.split(",")
                    for i, dataset_path in enumerate(datasets_paths):
                        evaluator = DatasetEvaluator(model_path, dataset_path, results_path / "random" / "epoch_{}".format(model_trainer._train_iteration))
                        metrics = evaluator.run() 
                        for k, v in metrics.items():
                            to_log["dataset{}_{}".format(i, k)]=v

                if cfg.evaluate.evaluate_model_accuracies:
                    eval_replay_buffer = mbrl.util.common.create_replay_buffer(
                                        cfg,
                                        obs_shape,
                                        act_shape,
                                        rng=rng,
                                        obs_type=dtype,
                                        action_type=dtype,
                                        reward_type=dtype,
                    )
                    eval_replay_buffer, evaluation_reward = evaluate(evaluation_env, copy.deepcopy(agent), number_episodes=cfg.evaluate.evaluate_number_episodes, replay_buffer=eval_replay_buffer)
                    to_log.update({"episode_reward": evaluation_reward[0], "episode_reward_ste": evaluation_reward[1]})
                    pathlib.Path.mkdir(model_path / "replay_buffers", parents=True, exist_ok=True)
                    if cfg.save_model_all_epochs:
                        torch.save(eval_replay_buffer, model_path / "replay_buffers" / "eval_{}.pt".format(model_trainer._train_iteration))
                    if cfg.save_replay_buffer_all_epochs:
                        torch.save(replay_buffer, model_path / "replay_buffers" / "train_{}.pt".format(model_trainer._train_iteration))

                logger.log_data(mbrl.constants.RESULTS_LOG_NAME, to_log)

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        current_trial += 1
        print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
