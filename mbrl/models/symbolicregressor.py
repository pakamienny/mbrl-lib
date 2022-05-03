# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import Not
from collections import defaultdict
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import sympy as sp
import sympytorch
from functools import partial
from mbrl.models.utils_symbolicregressor import *

from .model import Ensemble
import importlib
import os, sys


class MultiDimensionalRegressorWrapper(Ensemble):
    def __init__(self, regressor_class, regressor_args,  device, ensemble_size, **args):
        super().__init__(
            ensemble_size, device, propagation_method="random_model", deterministic=True
        )
        self.regressor_class, self.regressor_args = regressor_class, regressor_args
        for k,v in args.items():
            setattr(self, k, v)
        self.fake_param = nn.Parameter(torch.tensor(0.))
        self.models = None

    def reset_models(self):
        self.models = None

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        **args
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.models is None:
            return torch.zeros((x.shape[0], self.out_size)).to(self.device), None
        else:
            predicted = self.predict(x)
            return predicted[0], None

    def _fit(self, X, Y, regressor_class, idx=0):
        models=[]
        for dim in range(Y.shape[1]):
            self.regressor_args["random_state"]=idx
            regressor = regressor_class(**self.regressor_args)
            regressor.fit(X, Y[:,dim])
            if self.regressor_class == "operon.sklearn.SymbolicRegressor":
                model_function_str = regressor.get_model_string(3)
                model_function_str  = model_function_str.replace('^','**')
            else:
                replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/",
                                "abs": "Abs", "arctan": "atan"}
                model_function_str = regressor.retrieve_tree().infix()
                for op,replace_op in replace_ops.items():
                    model_function_str = model_function_str.replace(op,replace_op)
                model_function_str = model_function_str.replace("x_", "X")
                for i in range(X.shape[1]):
                    model_function_str = model_function_str.replace("X{}".format(i), "X{}".format(i+1))
            models.append(model_function_str)
        self.models[idx]=models

    def fit(self, X, Y):
        self.reset_models()
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(Y):
            Y = Y.cpu().numpy()
        models = getattr(self, "models", None)
        if models is None:
            print(sys.modules.keys())
            my_module, class_name = ".".join(self.regressor_class.split(".")[:-1]), self.regressor_class.split(".")[-1]
            my_module = importlib.import_module(my_module)
            regressor_class = getattr(my_module, class_name)
            self.models = defaultdict(list)
            for i in range(self.num_members):
                self._fit(X, Y, regressor_class=regressor_class, idx=i)
        self.compile_models()
        self.print_models()

    def compile_models(self):
        self.compiled_models = defaultdict(list)
        for idx in self.models:
            for model in self.models[idx]:
                model_sp = sp.parse_expr(model)
                model_numexpr = expr_to_numexpr_fn(model_sp)
                self.compiled_models[idx].append(model_numexpr)
        print("compiled models")

    def print_models(self, names=None):
        models_with_names = defaultdict(list)
        for idx in self.models:
            for model in self.models[idx]:
                model_with_name = copy.deepcopy(model)
                if names is not None:
                    for i, name in enumerate(names):
                        model_with_name = model_with_name.replace("X{}".format(i+1), name)
                    model_with_name = model_with_name.replace("X{}".format(len(names)+1), "reward")
                models_with_names[idx].append(model_with_name)
                print(sp.parse_expr(model_with_name))
        

    def update( 
        self,
        model_in,
        optimizer,
        target, 
        **args
    ):
        self.fit(model_in, target)
        loss, meta = self.loss(model_in, target)
        return loss.item(), meta

    def predict(self, X):
        Ys = []
        for idx in self.compiled_models:
            Y = []
            for model in self.compiled_models[idx]:
                Y.append(torch.tensor(model(X)).to(self.device))
            Y = torch.cat(Y,-1).unsqueeze(0)
            Ys.append(Y)
        Ys = torch.cat(Ys,0)
        return Ys



    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        return self._mse_loss(model_in, target), {}

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        #if model_in.ndim == 2:  # add model dimension
            #model_in = model_in.unsqueeze(0)
            #target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in)
        #return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()
        return F.mse_loss(pred_mean, target, reduction="none").sum(1).sum()

    #def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #    assert model_in.ndim == target.ndim
    #    pred_mean, _ = self.forward(model_in)
    #    print("pred", pred_mean.shape)
    #    print(target.shape)
    #    return F.mse_loss(pred_mean, target, reduction="none").sum(-1).mean()

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self._mse_loss(model_in, target), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "models": self.models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.models = model_dict["models"]
        self.compile_models()
