import torch
import torch.nn.functional as F
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.nn import Predictor, PredictorRegistry
from chemprop.nn.metrics import MSE, ChempropMetric
from chemprop.nn.predictors import MLP
from chemprop.nn.transforms import UnscaleTransform
from chemprop.utils import Factory
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor, nn


@PredictorRegistry.register("regression-moe")
class MixtureOfExpertsRegressionFFN(Predictor, HyperparametersMixin):
    r"""
    Implementation of the Adaptive Mixture of Local Experts [1] model for regression tasks.
    The works by passing the learned representation from message passing into one "gating network"
    and a configurable number of "experts". The outputs of the individual experts are
    multiplied element-wise by the output of the gating network, enabling the overall
    architecture to 'specialize' experts in certain types of inputs dynamically during
    training.

    References
    ----------
    .. [1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan and G. E. Hinton, "Adaptive Mixtures of Local Experts"
        Neural Computation, vol. 3, no. 1, pp. 79-87, March 1991, doi: 10.1162/neco.1991.3.1.79.
    """
    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_experts: int = 2,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
        gate_hidden_dim: int | None = None,
        gate_n_layers: int = 1,
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        """Adaptive Mixture of Local Experts Regression network

        Args:
            n_experts (int, optional): Number of expert sub-networks. Defaults to 2.
            n_tasks (int, optional): Output dimension for each sub-network. Defaults to 1.
            input_dim (int, optional): Size of message passing output. Defaults to DEFAULT_HIDDEN_DIM.
            hidden_dim (int, optional): Number of neurons per layer in sub-networks. Defaults to 300.
            n_layers (int, optional): Number of layers per network in sub-networks. Defaults to 1.
            dropout (float, optional): Dropout rate in all networks. Defaults to 0.0.
            activation (str | nn.Module, optional): Choice of activation function for all network. Defaults to "relu".
            gate_hidden_dim (int | None, optional): Number of neurons in gating network. Defaults to None.
            gate_n_layers (int, optional): Number of layers in gating network. Defaults to 1.
            criterion (ChempropMetric | None, optional): Criterion for training. Defaults to None.
            task_weights (Tensor | None, optional): Weights for each individual task. Defaults to None.
            threshold (float | None, optional): Passed to criterion. Defaults to None.
            output_transform (UnscaleTransform | None, optional): Output transform to be applied after forward. Defaults to None.
        """
        super().__init__()
        ignore_list = ["criterion", "output_transform", "activation"]
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__

        self.n_experts = n_experts

        # Experts
        self.experts = nn.ModuleList(
            [
                MLP.build(
                    input_dim,
                    n_tasks * self.n_targets,
                    hidden_dim,
                    n_layers,
                    dropout,
                    activation,
                )
                for _ in range(n_experts)
            ]
        )

        # Gating network
        gate_hidden_dim = gate_hidden_dim or hidden_dim
        self.gate = MLP.build(
            input_dim,
            n_experts,
            gate_hidden_dim,
            gate_n_layers,
            dropout,
            activation,
        )

        # Criterion
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        self.criterion = criterion or Factory.build(
            self._T_default_criterion, task_weights=task_weights, threshold=threshold
        )

        self.output_transform = (
            output_transform if output_transform is not None else nn.Identity()
        )

    @property
    def input_dim(self) -> int:
        return self.experts[0].input_dim

    @property
    def output_dim(self) -> int:
        return self.experts[0].output_dim

    @property
    def n_tasks(self) -> int:
        return self.output_dim // self.n_targets

    def forward(self, Z: Tensor) -> Tensor:
        expert_outputs = torch.stack(
            [expert(Z) for expert in self.experts], dim=1
        )  # [B, E, O]
        gate_logits = self.gate(Z)  # [B, E]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, E]

        Y = torch.einsum("be,bed->bd", gate_weights, expert_outputs)
        return self.output_transform(Y)

    train_step = forward

    def encode(self, Z: Tensor, i: int) -> Tensor:
        return self.experts[0][:i](Z)
