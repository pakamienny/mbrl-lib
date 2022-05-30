# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from posixpath import dirname
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy
import importlib
import warnings
import hydra
import omegaconf
import sklearn
from sklearn import ensemble
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn as nn
from torch.nn import functional as F
import mbrl.util.math
import sympy as sp
from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init
import numexpr as ne
from functools import partial
import operon

def get_method_type(regressor):
    if isinstance(regressor, getattr(importlib.import_module("sklearn.neural_network"), "MLPRegressor")):
        return "mlp"
    elif isinstance(regressor, getattr(importlib.import_module("operon.sklearn"), "SymbolicRegressor")):
        return "operon"
    elif isinstance(regressor, getattr(importlib.import_module("sklearn.gaussian_process"), "GaussianProcessRegressor")):
        return "GP"
    elif isinstance(regressor, getattr(importlib.import_module("sklearn.preprocessing"), "StandardScaler")):
        return "scaler"
    else:
        raise NotImplementedError

def expr_to_numexpr_fn(expr):
    infix = str(expr)
    
    def get_vals(dim, val):
        vals_ar = np.empty((dim,))
        vals_ar[:] = val
        return vals_ar

    def wrapped_numexpr_fn(_infix, x):
        if torch.is_tensor(x) and x.device != "cpu":
            x=x.cpu()
        local_dict = {}
        for d in range(x.shape[1]):
            if "X{}".format(d+1) in _infix:
                local_dict["X{}".format(d+1)]=x[:,d]
        try:
            vals = ne.evaluate(_infix, local_dict=local_dict)
        except Exception as e:
            return None
        if len(vals.shape)==0:
            vals = get_vals(x.shape[0], vals)
        return vals[:, None]
    return partial(wrapped_numexpr_fn, infix)

def get_model_sympy_expr(model):
    model_str = model.get_model_string(3).replace('^','**')
    return str(sp.parse_expr(model_str))


class Regressor():
    def __init__(self, regressor,  scale_x=False):
        self.regressor = regressor
        self.compiled_regressor = None
        self.str_regressor = None
        self.compiled_scale_x, self.str_scale_x = None, None
        if scale_x:  self.scale_x=StandardScaler()
        else: self.scale_x=None

    def __str__(self):
        if get_method_type(self.regressor) == "GP":
            return "GP"
        return self.str_regressor

    def __repr__(self):
        if get_method_type(self.regressor) == "GP":
            return "GP"
        return self.str_regressor

    def fit(self, X, y):
        if self.scale_x is not None:
            scaled_X = self.scale_x.fit_transform(X)
        else:
            scaled_X = X
        self.regressor.fit(scaled_X, y)

        print("error: ", ((self.regressor.predict(scaled_X)- y)**2).max())

    def forward(self, X):
        if self.compiled_regressor is None:
            return (None, None)

        if self.compiled_scale_x is not None:
            scaled_X = self.compiled_scale_x.transform(X)
        else:
            scaled_X = X

        if get_method_type(self.regressor) == "GP": 
            #with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            y_tilde, y_tilde_std = self.compiled_regressor.predict(scaled_X, return_std=True)
        else:
            raise NotImplementedError
        return (y_tilde, np.log(1e-7+np.sqrt(y_tilde_std)))

    def export_str(self):
        state_dict = {}
        for regr in ["regressor", "scale_x"]:
            state_dict[regr] = getattr(self,  "str_"+regr)
        return state_dict

    def import_str(self, state_dict):
        for regr in ["regressor", "scale_x"]:
            if regr in state_dict:
                setattr(self,  "str_"+regr, state_dict[regr])
        self.compile()

    def compile_str(self):
        for regr in ["regressor", "scale_x"]:
            regressor = getattr(self, regr)
            if regressor is None: 
                continue
            setattr(self, "str_"+regr,  regressor)

    def compile(self):
        for regr in ["regressor", "scale_x"]:
            regressor = getattr(self, regr)
            str_regressor = getattr(self, "str_"+regr)
            if regressor is None: 
                continue
            setattr(self, "compiled_"+regr,  str_regressor)
            
class StackRegressorWithVariance():
    def __init__(self, regressor, regressor_variance=None, scale_x=False, scale_y=False):
        self.regressor = regressor
        self.regressor_variance=regressor_variance
        self.compiled_regressor, self.compiled_regressor_variance = None, None
        self.str_regressor, self.str_regressor_variance = None, None
        self.compiled_scale_x, self.compiled_scale_y = None, None
        self.str_scale_x, self.str_scale_y = None, None
        if scale_x:  self.scale_x=StandardScaler()
        else: self.scale_x=None
        if scale_y:  self.scale_y=StandardScaler()
        else: self.scale_y=None

    def __str__(self):
        if get_method_type(self.regressor) == "mlp":
            return "MLP"
        if get_method_type(self.regressor) == "GP":
            return "GP"
        return self.str_regressor if self.str_regressor is not None else "NaN"

    def __repr__(self):
        if get_method_type(self.regressor) == "mlp":
            return "MLP"
        if get_method_type(self.regressor) == "GP":
            return "GP"
        return self.str_regressor if self.str_regressor is not None else "NaN"

    def fit(self, X, y):
        if self.scale_x is not None:
            scaled_X = self.scale_x.fit_transform(X)
        else:
            scaled_X = X
        if self.scale_y is not None:
            scaled_y = self.scale_y.fit_transform(y[:,None])[:,0]
        else:
            scaled_y = y
        self.regressor.fit(scaled_X, scaled_y)
        if self.regressor_variance is not None:
            y_tilde = self.regressor.predict(scaled_X)
            self.regressor_variance.fit(scaled_X,(scaled_y-y_tilde)**2)
    
    def forward(self, X):
        return (self.forward_mean(X), self.forward_logvar(X))

    def forward_mean(self, X):
        if self.compiled_regressor is None:
            return None

        if self.compiled_scale_x is not None:
            scaled_X = self.compiled_scale_x.transform(X)
        else:
            scaled_X = X

        if get_method_type(self.regressor) in ["mlp", "GP"]: 
            y_tilde = self.compiled_regressor.predict(scaled_X)
        else:
            y_tilde = self.compiled_regressor(scaled_X)
        
        if self.compiled_scale_y is not None:
            unscaled_y = self.compiled_scale_y.inverse_transform(y_tilde)
        else:
            unscaled_y = y_tilde
        return unscaled_y

    def forward_logvar(self, X):
        if self.compiled_regressor_variance is None:
            return None
        var = self.compiled_regressor_variance.predict(X)
        logvar = np.log(np.abs(var)+1e-7)
        return logvar

    def export_str(self):
        state_dict = {}
        for regr in ["regressor", "regressor_variance", "scale_x", "scale_y"]:
            state_dict[regr] = getattr(self,  "str_"+regr)
        return state_dict

    def import_str(self, state_dict):
        for regr in ["regressor", "regressor_variance", "scale_x", "scale_y"]:
            if regr in state_dict:
                setattr(self,  "str_"+regr, state_dict[regr])
        self.compile()

    def compile_str(self):
        for regr in ["regressor", "regressor_variance","scale_x", "scale_y"]:
            regressor = getattr(self, regr)
            if regressor is None: 
                continue
            if get_method_type(regressor) == "operon":
                try:
                    setattr(self, "str_"+regr,  get_model_sympy_expr(regressor))
                except TypeError as e:
                    print(e)
                    setattr(self, "str_"+regr,  None)
            else:
                setattr(self, "str_"+regr,  regressor)

    def compile(self):
        for regr in ["regressor", "regressor_variance", "scale_x", "scale_y"]:
            regressor = getattr(self, regr)
            str_regressor = getattr(self, "str_"+regr)
            if regressor is None: 
                continue
            if get_method_type(regressor) == "operon":
                try:
                    setattr(self, "compiled_"+regr,  expr_to_numexpr_fn(str_regressor) if str_regressor is not None else None)
                except TypeError as e:
                    setattr(self, "compiled_"+regr, None)
            else:
                setattr(self, "compiled_"+regr,  str_regressor)

def create_model_with_variance(base_regressor_cfg, random_state, deterministic=False, scale_x=False, scale_y=False):
    regressor = hydra.utils.instantiate(base_regressor_cfg, random_state=random_state)
    if deterministic:
        regressor_variance = None
    else:
        regressor_variance = GaussianProcessRegressor(normalize_y=True)
    return StackRegressorWithVariance(regressor, regressor_variance, scale_x=scale_x, scale_y=scale_y)

def create_gp_model(base_regressor_cfg, random_state, deterministic=False, scale_x=False, scale_y=False):
    regressor = hydra.utils.instantiate(base_regressor_cfg, random_state=random_state, normalize_y=scale_y)
    return Regressor(regressor, scale_x=scale_x)


class SymbolicRegressorMatrix(nn.Module):
    def __init__(self, base_regressor_cfg, ensemble_size, out_size, deterministic, device, scale_x=False, scale_y=False, **args):
        super().__init__()

        """For now, elite is defined for all dimensions of output"""
        self.deterministic=deterministic
        self.base_regressor_cfg=base_regressor_cfg
        if base_regressor_cfg._target_ == "sklearn.gaussian_process.GaussianProcessRegressor":
            self.models=[
                [
                    create_gp_model(self.base_regressor_cfg, e, deterministic, scale_x, scale_y) for o in range(out_size)
                ] 
                    for e in range(ensemble_size)
                ]
        else:
            self.models=[
                [
                    create_model_with_variance(self.base_regressor_cfg, e, deterministic, scale_x, scale_y) for o in range(out_size)
                ] 
                    for e in range(ensemble_size)
                ]
        self.out_size=out_size
        self.elite_models: List[int] = None
        self.use_only_elite = False
        self.fake_param = nn.Parameter(torch.tensor(0.))
        self.device=device

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite

    def set_elite(self, elite_indices: Sequence[int]):
        self.elite_models = list(elite_indices)

    def compile(self):
        for model in self.models:
            for dim in range(self.out_size):
                model[dim].compile()

    def compile_str(self):
        for model in self.models:
            for dim in range(self.out_size):
                model[dim].compile_str()

    def export_str(self):
        state_dicts = [[model[dim].export_str() for dim in range(self.out_size)] for model in self.models]
        return state_dicts

    def import_str(self, states_dict):
        state_dicts = [[self.models[model_id][dim].import_str(states_dict[model_id][dim]) for dim in range(self.out_size)] for model_id in range(len(self.models))]
        return state_dicts

    def __str__(self):
        for model_id, model in enumerate(self.models): 
            for dim in range(self.out_size):
                print("model id: {}, dim {}, expr: {} ".format(model_id, dim, str(model[dim])))

    def __repr__(self):
        repr=""
        for model_id, model in enumerate(self.models): 
            for dim in range(self.out_size):
                repr+="model id: {}, dim {}, expr: {} \n".format(model_id,  dim, str(model[dim]))
        return repr
                
    def fit(self, X, y, max_trials=10):
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        #assert X_np.ndim==3 and y_np.shape[0]==self.num_members, "problem with shape when fitting {}, {}".format(X_np.shape, y_np.shape)
        for model_id, model in enumerate(self.models): 
            for dim in range(self.out_size):
                trial = 0
                while trial<max_trials:
                    model[dim].fit(X_np[model_id], y_np[model_id, :, dim])
                    if get_method_type(model[dim].regressor) == "operon":
                        expr = get_model_sympy_expr(model[dim].regressor)
                        if expr == sp.nan:
                            print("nan detected")
                            trial+=1
                            model[dim] = create_model_with_variance(self.base_regressor_cfg, model_id+1000, self.deterministic)
                        else:
                            break
                    else:
                        break
        self.compile_str()
        self.compile()
    
    def forward(self, x):
        x_np = x.cpu().numpy()
        if x_np.ndim == 3 and x_np.shape[0]:
            x_np = x_np[0]

        means = []
        logvars = None if self.deterministic else []
        to_forward = self.elite_models if self.use_only_elite else [i for i in range(len(self.models))]
        models = [model for model_id, model in enumerate(self.models) if model_id in to_forward]
        if x_np.ndim == 3:
            for model_id, model in enumerate(models):
                means_i, logvars_i = [], []
                for j in range(self.out_size):
                    y_tilde, logvar = None, None
                    if model[j] is not None:
                        y_tilde, logvar = model[j].forward(x_np[model_id])
                    if y_tilde is None:
                        y_tilde = np.zeros((x_np.shape[1], 1))
                    else:
                        if y_tilde.ndim==1:
                            y_tilde = np.expand_dims(y_tilde, -1)
                    means_i.append(y_tilde)
                    if not self.deterministic:
                        if logvar is None:
                            logvar = np.zeros((x_np.shape[1], 1))  
                        if logvar.ndim==1:
                            logvar = np.expand_dims(logvar, -1)
                        logvars_i.append(logvar)

                means_i = np.concatenate(means_i, -1)
                means.append(means_i[None, :])
                if not self.deterministic:
                    logvars_i = np.concatenate(logvars_i, -1)
                    logvars.append(logvars_i[None, :])
            means = np.concatenate(means, 0)
            means = torch.tensor(means).to(self.device)
            if not self.deterministic:
                logvars = np.concatenate(logvars, 0)
                logvars = torch.tensor(logvars).to(self.device)

        else:
            for model_id, model in enumerate(models):
                means_i, logvars_i = [], []
                for j in range(self.out_size):
                    y_tilde, logvar = None, None
                    if model[j] is not None:
                        y_tilde, logvar = model[j].forward(x_np)
                    if y_tilde is None:
                        y_tilde = np.zeros((x_np.shape[0], 1))
                    if y_tilde.ndim==1:
                        y_tilde = np.expand_dims(y_tilde, -1)
                    means_i.append(y_tilde)
                    if not self.deterministic:
                        if logvar is None:
                            logvar = np.zeros((x_np.shape[0], 1))
                        if logvar.ndim==1:
                            logvar = np.expand_dims(logvar, -1)
                        logvars_i.append(logvar)
                means_i = np.concatenate(means_i, -1)
                means.append(means_i[None, :])
                if not self.deterministic:
                    logvars_i = np.concatenate(logvars_i, -1)
                    logvars.append(logvars_i[None, :])
            means = np.concatenate(means, 0)
            means = torch.tensor(means).to(self.device)
            if not self.deterministic:
                logvars = np.concatenate(logvars, 0)
                logvars = torch.tensor(logvars).to(self.device)
        return means, logvars
        
