# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from posixpath import dirname
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy

import hydra
import omegaconf
import sklearn
from sklearn import ensemble
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F
import mbrl.util.math
import sympy as sp
from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init
import numexpr as ne
from functools import partial

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
   
def get_model_sympy_expr(model):
    model_str = model.get_model_string(3).replace('^','**')
    return sp.parse_expr(model_str) 

class StackedModels(nn.Module):
    def __init__(self, ensemble_size, out_size, deterministic, device):
        super().__init__()

        """For now, elite is defined for all dimensions of output"""
        self.deterministic=deterministic
        self.models=[[create_operon_model(e, deterministic) for o in range(out_size)] for e in range(ensemble_size)]
        self.num_members=ensemble_size
        self.out_size=out_size
        self.elite_models: List[int] = None
        self.use_only_elite = False
        self.fake_param = nn.Parameter(torch.tensor(0.))
        self.device=device
        self.string_models = [[None for _ in range(self.out_size)] for _ in range(self.num_members)]
        self.compiled_models = [[None for _ in range(self.out_size)] for _ in range(self.num_members)]

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)
   
    def compile_string(self):
        for model_id, model in enumerate(self.models):
            for dim in range(self.out_size):
                try:
                    expr = get_model_sympy_expr(model[dim].regressor)
                except Exception as e:
                    break
                if expr != sp.nan:
                    self.string_models[model_id][dim]=(expr, model[dim].regressor_variance)
                else:
                    print("failed dim: {}".format(dim))

    def compile(self):
        for model_id, model in enumerate(self.string_models):
            for dim in range(self.out_size):
                self.compiled_models[model_id][dim]=(expr_to_numexpr_fn(model[dim][0]), model[dim][1])

    def __str__(self):
        for model_id, model in enumerate(self.string_models): 
            for dim in range(self.out_size):
                print("model id: {}, dim {}, expr: {} ".format(model_id, dim, model[dim][0]))

    def __repr__(self):
        repr=""
        for model_id, model in enumerate(self.string_models): 
            for dim in range(self.out_size):
                repr+="model id: {}, dim {}, expr: {} \n".format(model_id,  dim, model[dim][0])
        return repr
                
    def fit(self, X, y, max_trials=10):
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        assert X_np.ndim==3 and y_np.shape[0]==self.num_members, "problem with shape when fitting {}".format(X_np.shape, y_np.shape)
        for model_id, model in enumerate(self.models): 
            for dim in range(self.out_size):
                trial = 0
                while trial<max_trials:
                    model[dim].fit(X_np[model_id], y_np[model_id, :, dim])
                    expr = get_model_sympy_expr(model[dim].regressor)
                    if expr == sp.nan:
                        trial+=1
                        model[dim] = create_operon_model(model_id+1000, self.deterministic)
                    else:
                        break
        self.compile_string()
        self.compile()

    def forward(self, x):
        x_np = x.cpu().numpy()
        if x_np.ndim == 3 and x_np.shape[0]:
            x_np = x_np[0]

        means, logvars = [], []
        to_forward = self.elite_models if self.use_only_elite else [i for i in range(self.num_members)]
        compiled_models = [model for model_id, model in enumerate(self.compiled_models) if model_id in to_forward]
        models = [model for model_id, model in enumerate(self.models) if model_id in to_forward]

        if x_np.ndim == 3:
            for model_id, model in enumerate(compiled_models):
                means_i, logvars_i = [], []
                for j in range(self.out_size):
                    if model[j] == None:
                        y_tilde, logvar = np.zeros((x_np.shape[1], 1)), np.ones((x_np.shape[1], 1))
                    else:
                        y_tilde = model[j][0](x_np[model_id])
                        logvar = models[model_id][j].forward_logvar(x_np[model_id])
                        if y_tilde.ndim==1:
                            y_tilde = np.expand_dims(y_tilde, -1)
                        if logvar.ndim==1:
                            logvar = np.expand_dims(logvar, -1)
                    means_i.append(y_tilde)
                    logvars_i.append(logvar)
                means_i = np.concatenate(means_i, -1)
                means.append(means_i[None, :])
                logvars_i = np.concatenate(logvars_i, -1)
                logvars.append(logvars_i[None, :])
            means = np.concatenate(means, 0)
            logvars = np.concatenate(logvars, 0)
            means = torch.tensor(means).to(self.device)
            logvars = torch.tensor(logvars).to(self.device)

        else:
            for model_id, model in enumerate(compiled_models):
                means_i, logvars_i = [], []
                for j in range(self.out_size):
                    if model[j] == None:
                        y_tilde, logvar = np.zeros((x_np.shape[0], 1)), np.ones((x_np.shape[0], 1))
                    else:
                        y_tilde = model[j][0](x_np)
                        logvar = models[model_id][j].forward_logvar(x_np)
                        if y_tilde.ndim==1:
                            y_tilde = np.expand_dims(y_tilde, -1)
                        if logvar.ndim==1:
                            logvar = np.expand_dims(logvar, -1)
                    means_i.append(y_tilde)
                    logvars_i.append(logvar)
                means_i = np.concatenate(means_i, -1)
                means.append(means_i[None, :])
                logvars_i = np.concatenate(logvars_i, -1)
                logvars.append(logvars_i[None, :])
            means = np.concatenate(means, 0)
            logvars = np.concatenate(logvars, 0)
            means = torch.tensor(means).to(self.device)
            logvars = torch.tensor(logvars).to(self.device)
        return means, logvars
        
class StackRegressorWithVariance():
    def __init__(self, regressor, deterministic=False, max_logvar=0.5, min_logvar=-10):
        self.regressor = regressor
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar

        if not deterministic:
            self.regressor_variance = None
        else:
            from sklearn.gaussian_process import GaussianProcessRegressor
            self.regressor_variance = GaussianProcessRegressor(normalize_y=True)
   

    def fit(self, X, y):
        self.regressor.fit(X,y)
        if self.regressor_variance is not None:
            y_tilde = self.regressor.predict(X)
            self.regressor_variance.fit(X,(y-y_tilde)**2)
    
    def forward_logvar(self, X):

        def softplus(x, beta=1., threshold=20.):
            out = copy.deepcopy(x)
            mask=x*beta<=threshold
            out[mask]=1/beta*np.log(1+np.exp(beta*x[mask]))
            return out

        if self.regressor_variance is None:
            return np.ones(X.shape[0])
        var = self.regressor_variance.predict(X)
        logvar = np.log(np.abs(var)+1e-7)
        logvar = self.max_logvar - softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + softplus(logvar - self.min_logvar)
        return logvar

def create_operon_model(random_state, deterministic=False):
    from operon.sklearn import SymbolicRegressor
    regr = SymbolicRegressor(
                local_iterations=5,
                generations= 10000,
                n_threads= 10,
                random_state=random_state,
                time_limit= 240,
                max_evaluations= 500000,
                population_size= 5000,
                allowed_symbols= 'add,sub,mul,div,constant,variable,cos,sin,pow,exp',     
                objectives= ["r2"],
                reinserter='keep-best'
            )
    
    return StackRegressorWithVariance(regr, deterministic)



class Operon(Ensemble):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution.

    This model corresponds to a Probabilistic Ensemble in the Chua et al.,
    NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        learn_logvar_bounds (bool): if ``True``, the logvar bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ReLU when ``None``.
    """


    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        ensemble_size: int = 1,
        deterministic: bool = True,
        propagation_method: Optional[str] = None,
        **args

    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )
        for k,v in args.items():
            setattr(self, k, v)

        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.model = StackedModels(ensemble_size, out_size, deterministic, device)
        self.elite_models: List[int] = None
        self.min_logvar = -10 
        self.max_logvar = 0.5 

    def get_string_models(self):
        return self.model.string_models

    def set_string_models(self, string_models):
        self.model.string_models=string_models

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            self.model.set_elite(self.elite_models)
            self.model.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        mean, logvar = self.model(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        return mean, logvar

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation. 
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def fit(self, model_in: torch.Tensor, target: torch.Tensor):
        self.model.fit(model_in, target)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim

        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()


    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar - self.min_logvar)
        return nll

    def update(
        self,
        model_in,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Updates the model using backpropagation with given input and target tensors.

        Provides a basic update function, following the steps below:

        .. code-block:: python

           optimizer.zero_grad()
           loss = self.loss(model_in, target)
           loss.backward()
           optimizer.step()

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            optimizer (torch.optimizer): the optimizer to use for the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
             (float): the numeric value of the computed loss.
             (dict): any additional metadata dictionary computed by :meth:`loss`.
        """
        loss, meta = self.loss(model_in, target, fit=True)
        return loss.item(), meta

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        fit: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if fit:
            self.fit(model_in, target)

        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        else:
            return self._nll_loss(model_in, target), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

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

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "string_models": self.get_string_models(),
            "elite_models": self.elite_models,

        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.elite_models = model_dict["elite_models"]
        self.set_string_models(model_dict["string_models"])
        self.model.compile()