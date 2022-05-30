# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pathlib
from typing import List

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

import mbrl.util
import mbrl.util.common
from sklearn.metrics import r2_score, mean_squared_error
import torch
import hydra

class RewardEvaluator:
    def __init__(self, model_dir: str):
        self.model_dir = pathlib.Path(model_dir)
        pathlib.Path.mkdir(self.model_dir, parents=True, exist_ok=True)

        self.cfg = mbrl.util.common.load_hydra_cfg(self.model_dir)
        self.handler = mbrl.util.create_handler(self.cfg)
        torch_generator = torch.Generator(device=self.cfg.device)

        self.env, self.term_fn, self.reward_fn = self.handler.make_env(self.cfg)
        self.dynamics_model = mbrl.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_dir,
        )

        model_env = mbrl.models.ModelEnv(
            self.env, self.dynamics_model,  self.term_fn, self.reward_fn, generator=torch_generator
        )
         
        if  self.cfg.algorithm.agent == "mbrl.planning.RandomAgent":
            self.agent = mbrl.planning.RandomAgent(self.env)
        elif self.cfg.algorithm.agent._target_ == "mbrl.third_party.pytorch_sac_pranz24.sac.SAC":
            from mbrl.planning.sac_wrapper import SACAgent
            import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
            from typing import Optional, Sequence, cast
            mbrl.planning.complete_agent_cfg(self.env, self.cfg.algorithm.agent)

            self.agent = SACAgent(
                cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(self.cfg.algorithm.agent, _recursive_=False))
            )

        else:
            self.agent = mbrl.planning.create_trajectory_optim_agent_for_model(
                model_env, self.cfg.algorithm.agent, num_particles=self.cfg.algorithm.num_particles
            )

    def run(self, num_episodes):

        replay_buffer = mbrl.util.common.create_replay_buffer(
            mbrl.util.common.load_hydra_cfg(self.model_dir),
            self.env.observation_space.shape,
            self.env.action_space.shape,
        )
        rewards = mbrl.util.common.rollout_agent_trajectories(
            self.env,
            agent=self.agent,
            agent_kwargs={},
            steps_or_trials_to_collect=num_episodes,
            collect_full_trajectories=True,
            replay_buffer=replay_buffer
        )
        return replay_buffer,  (np.mean(rewards), np.std(rewards)/np.sqrt(num_episodes))

class DatasetEvaluator:
    def __init__(self, model_dir: str, dataset_dir: str, output_dir: str):
        self.model_path = pathlib.Path(model_dir)
        self.output_path = pathlib.Path(output_dir)
        pathlib.Path.mkdir(self.output_path, parents=True, exist_ok=True)

        self.cfg = mbrl.util.common.load_hydra_cfg(self.model_path)
        self.handler = mbrl.util.create_handler(self.cfg)

        self.env, term_fn, reward_fn = self.handler.make_env(self.cfg)
        self.reward_fn = reward_fn
        self.dynamics_model = mbrl.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_path,
        )

        self.replay_buffer = mbrl.util.common.create_replay_buffer(
            mbrl.util.common.load_hydra_cfg(dataset_dir),
            self.env.observation_space.shape,
            self.env.action_space.shape,
            load_dir=dataset_dir,
        )

    def compute_metrics(self, pred, target, metrics="r2,mse"):
        assert pred.ndim==target.ndim==2
        
        out_size = pred.shape[1]
        results = {}
        for dim in range(out_size) :
            if "r2" in metrics.split(","):
                r2 = r2_score(target[:, dim], pred[:, dim])
                results["r2_dim{}".format(dim)]=r2
            if "mse" in metrics.split(","):
                mse = mean_squared_error(target[:, dim], pred[:, dim])
                results["mse_dim{}".format(dim)]=mse
        return results

    def plot_dataset_results(self, dataset: mbrl.util.TransitionIterator, compute_stats: bool = True):
        all_means: List[np.ndarray] = []
        all_targets = []

        # Iterating over dataset and computing predictions
        for batch in dataset:
            (
                outputs,
                target,
            ) = self.dynamics_model.get_output_and_targets(batch)
            all_means.append(outputs[0].cpu().numpy())
            all_targets.append(target.cpu().numpy())

        # Consolidating targets and predictions
        all_means_np = np.concatenate(all_means, axis=-2)
        targets_np = np.concatenate(all_targets, axis=0)

        if all_means_np.ndim == 2:
            all_means_np = all_means_np[np.newaxis, :]
        assert all_means_np.ndim == 3  # ensemble, batch, target_dim

        
        metrics = {}
        if compute_stats:
            ensemble_size = all_means_np.shape[0]
            for ensemble in range(ensemble_size):
                metrics_ensemble=self.compute_metrics(all_means_np[ensemble], targets_np)
                for k,v in metrics_ensemble.items():
                    metrics[k+"_ens{}".format(ensemble)]=v
       
        # Visualization
        num_dim = targets_np.shape[1]
        for dim in range(num_dim):
            sort_idx = targets_np[:, dim].argsort()
            subsample_size = len(sort_idx) // 20 + 1
            subsample = np.random.choice(len(sort_idx), size=(subsample_size,))
            means = all_means_np[..., sort_idx, dim][..., subsample]  # type: ignore
            target = targets_np[sort_idx, dim][subsample]

            plt.figure(figsize=(8, 8))
            for i in range(all_means_np.shape[0]):
                plt.plot(target, means[i], ".", markersize=2)
            mean_of_means = means.mean(0)
            mean_sort_idx = target.argsort()
            plt.plot(
                target[mean_sort_idx],
                mean_of_means[mean_sort_idx],
                color="r",
                linewidth=0.5,
            )
            plt.plot(
                [target.min(), target.max()],
                [target.min(), target.max()],
                linewidth=2,
                color="k",
            )
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            fname = self.output_path / f"pred_dim{dim}.png"
            plt.savefig(fname)
            plt.close()
        return metrics

    def run(self, compute_stats=True):
        batch_size = -1
        if hasattr(self.dynamics_model, "set_propagation_method"):
            self.dynamics_model.set_propagation_method(None)
            # Some models (e.g., GaussianMLP) require the batch size to be
            # a multiple of number of models
            batch_size = len(self.dynamics_model) * 8
        dataset, _ = mbrl.util.common.get_basic_buffer_iterators(
            self.replay_buffer, batch_size=batch_size, val_ratio=0
        )
        return self.plot_dataset_results(dataset, compute_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    if not args.dataset_dir:
        args.dataset_dir = args.model_dir
    evaluator = DatasetEvaluator(args.model_dir, args.dataset_dir, args.results_dir)

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["font.size"] = 14

    evaluator.run(compute_stats=False)
