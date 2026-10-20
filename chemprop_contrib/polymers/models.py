from __future__ import annotations

import logging

import torch
from torch import Tensor

from chemprop.models.model import MPNN
from chemprop.nn.agg import MeanAggregation
from chemprop_contrib.polymers.data import BatchWeightedMolGraph

logger = logging.getLogger(__name__)


def _weighted_mean(H: Tensor, batch: Tensor, V_w: Tensor, dim: int = 0) -> Tensor:
    r"""Average the graph-level representation, weighted by the atom weights:

    .. math::
        \mathbf h = \frac{1}{\sum_{v \in V} w_v} \sum_{v \in V} \mathbf h_v

    :class:`WeightedBondMessagePassing` has already multiplied :math:`\mathbf h_v` by
    :math:`w_v`. This is the weighted branch the original PR added to
    :meth:`chemprop.nn.agg.MeanAggregation.forward`.
    """
    dim_size = batch.max().int() + 1
    index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
    sum_node_embeddings = torch.zeros(
        dim_size, H.shape[1], dtype=H.dtype, device=H.device
    ).scatter_reduce_(dim, index_torch, H, reduce="sum", include_self=False)
    V_w = V_w.unsqueeze(1)
    index_weight = batch.unsqueeze(1).repeat(1, V_w.shape[1])
    sum_of_weights = torch.zeros(
        dim_size, V_w.shape[1], dtype=V_w.dtype, device=V_w.device
    ).scatter_reduce_(dim, index_weight, V_w, reduce="sum", include_self=False)
    return torch.div(sum_node_embeddings, sum_of_weights)


class wMPNN(MPNN):
    r"""An :class:`wMPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine. Messages are weighted based on bond and atom weights. The final
    result is multiplied by a scaling factor (e.g. the degree of polymerisation).

    The first two modules calculate learned fingerprints from an input molecule or
    polymer graph, and the final module takes these learned fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{wMPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MessagePassing
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level prediction
    predictor : Predictor
        the function to use to calculate the final prediction
    batch_norm : bool, default=False
        if `True`, apply batch normalization to the output of the aggregation operation
    metrics : Iterable[Metric] | None, default=None
        the metrics to use to evaluate the model during training and evaluation
    warmup_epochs : int, default=2
        the number of epochs to use for the learning rate warmup
    init_lr : int, default=1e-4
        the initial learning rate
    max_lr : float, default=1e-3
        the maximum learning rate
    final_lr : float, default=1e-4
        the final learning rate

    Raises
    ------
    ValueError
        if the output dimension of the message passing block does not match the input dimension of
        the predictor function
    """

    def fingerprint(
        self, bmg: BatchWeightedMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        # chemprop's aggregations don't accept atom weights, so the weighted mean is computed
        # here. The other aggregations ignored V_w in the original PR.
        # NOTE:
        # - subclasses of MeanAggregation also take this branch, so their forward() is
        #   bypassed, as are any hooks registered on self.agg
        # - no aggregation receives V_w; a custom weight-aware aggregation needs changes here
        # - _weighted_mean is a copy of the PR's MeanAggregation logic and won't track
        #   upstream changes to MeanAggregation
        if isinstance(self.agg, MeanAggregation):
            H = _weighted_mean(H_v, bmg.batch, bmg.V_w, self.agg.dim)
        else:
            H = self.agg(H_v, bmg.batch)
        H = self.weight(H, bmg.degree_of_poly)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def weight(self, H: Tensor, degree_of_poly: Tensor) -> Tensor:
        """weights the final molecular representation by the degree of polymerization"""
        return torch.mul(degree_of_poly.unsqueeze(1), H)
