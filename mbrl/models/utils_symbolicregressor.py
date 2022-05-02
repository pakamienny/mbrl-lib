# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import Not
from collections import defaultdict
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy
import time

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import sympy as sp
import sympytorch
from functools import partial
import os

import numexpr as ne

def expr_to_torch_module(expr, device, dtype=torch.float32):
    mod = sympytorch.SymPyModule(expressions=[expr])
    mod = mod.to(device)
    mod = mod.to(dtype)
    def wrapper_fn(_mod, x):
        local_dict = {}
        if x.ndim==1:
            for d in range(x.shape[0]):
                local_dict["X{}".format(d+1)]=x[d]
        elif x.ndim==2:
            for d in range(x.shape[1]):
                local_dict["X{}".format(d+1)]=x[:,d]
        else:
            raise NotImplementedError
        y = _mod(**local_dict)
        if y.ndim==1:
            y = y.unsqueeze(0).repeat(x.shape[0],1)
        return y
    return partial(wrapper_fn, mod)

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
        vals = ne.evaluate(_infix, local_dict=local_dict)
        if len(vals.shape)==0:
            vals = get_vals(x.shape[0], vals)
        return vals[:, None]

    return partial(wrapped_numexpr_fn, infix)
    