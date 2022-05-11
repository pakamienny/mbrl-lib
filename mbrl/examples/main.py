# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import omegaconf



@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    import operon
    import torch
    import numpy as np
    import mbrl.algorithms.mbpo as mbpo
    import mbrl.algorithms.pets as pets
    import mbrl.algorithms.planet as planet
    import mbrl.util.env
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name in ["pets", "random"]:
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)


if __name__ == "__main__":
    run()
