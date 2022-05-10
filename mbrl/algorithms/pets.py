# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional
import hydra

import gym
import numpy as np
import omegaconf
import torch
import importlib
import copy

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.diagnostics.eval_model_on_dataset import DatasetEvaluator
import pathlib
import warnings
warnings.simplefilter("ignore")

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

def evaluate(
    env: gym.Env,
    agent: mbrl.planning.Agent,
    number_episodes: int = 10,
):
    agent.reset()
    rewards = mbrl.util.common.rollout_agent_trajectories(
        env,
        agent=agent,
        agent_kwargs={},
        steps_or_trials_to_collect=number_episodes,
        collect_full_trajectories=True
    )
    assert len(rewards)==number_episodes, "problem with number of rewards"
    return (np.mean(rewards), np.std(rewards)/np.sqrt(number_episodes))

def create_trainer(cfg, dynamics_model, logger, tensorboard_logger):

    model_trainer_type = cfg.algorithm.model_trainer
    if model_trainer_type == "SymbolicModelTrainer":
        model_trainer = mbrl.models.SymbolicModelTrainer(
                dynamics_model,
                logger=logger,
                tensorboard_logger=tensorboard_logger
            )
    else:
        model_trainer = mbrl.models.ModelTrainer(
            dynamics_model,
            optim_lr=cfg.overrides.model_lr,
            weight_decay=cfg.overrides.model_wd,
            logger=logger,
            tensorboard_logger=tensorboard_logger
            )     
    return model_trainer

def make_copies(*args):
    res = []
    for arg in args:
        print(arg)
        res.append(copy.deepcopy(arg))
    return tuple(res)

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
    #if cfg.overrides.get("obs_process_fn", None) is not None:
    #    obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    #    obs_shape = obs_process_fn(env.reset()).shape 
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    if work_dir is None:
        work_dir = os.getcwd()
    print(f"Results will be saved at {work_dir}.")
    if silent:
        logger = None
    else:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_logger = SummaryWriter(work_dir)
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    dynamics_model_model = copy.deepcopy(dynamics_model.model)
    model_trainer = create_trainer(cfg, dynamics_model, logger, tensorboard_logger)

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

    evaluation_reward_mean, evaluation_reward_ste = evaluate(evaluation_env, mbrl.planning.RandomAgent(env))
    logger.log_data(
        mbrl.constants.RESULTS_LOG_NAME,
        {"env_step": 0, 
        "episode_reward": evaluation_reward_mean,
        "episode_reward_ste": evaluation_reward_ste,
        "in_domain_accuracy": np.nan,
        "in_domain_accuracy_ste": np.nan,
        "random_accuracy": np.nan,
        "random_accuracy_ste": np.nan,
        },
        )
    print("random evaluation", evaluation_reward_mean)
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
    env_steps =  0
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
                if cfg.algorithm.reinitialize_models:
                    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape, model=dynamics_model_model)
                    prev_train_iteration = model_trainer._train_iteration
                    model_trainer = create_trainer(cfg, dynamics_model, logger, tensorboard_logger) 
                    model_trainer._train_iteration = prev_train_iteration

                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )
                to_log = {"env_step": cfg.algorithm.initial_exploration_steps+env_steps}

                if cfg.evaluate.evaluate_model_accuracies:
                    model_path = pathlib.Path(work_dir)
                    ##current buffer
                    results_path = model_path / "diagnostics" / "dataset"
                    evaluator = DatasetEvaluator(model_path, model_path, results_path / "in_domain" / "epoch_{}".format(model_trainer._train_iteration))
                    in_domain_accuracies = evaluator.run() 
                    to_log.update({"in_domain_accuracy": in_domain_accuracies[0], "in_domain_accuracy_ste": in_domain_accuracies[1]})

                    ##random_data   
                    evaluator = DatasetEvaluator(model_path, cfg.overrides.evaluate_dataset_path, results_path / "random" / "epoch_{}".format(model_trainer._train_iteration))
                    random_domain_accuracies = evaluator.run() 
                    to_log.update({"random_accuracy": random_domain_accuracies[0], "in_domain_accuracy_ste": random_domain_accuracies[1]})
                
                if cfg.evaluate.evaluate_all_model_updates:
                    evaluation_reward = evaluate(evaluation_env, copy.deepcopy(agent), number_episodes=cfg.evaluate.evaluate_number_episodes)
                    to_log.update({"episode_reward": evaluation_reward[0], "episode_reward_ste": evaluation_reward[1]})

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
