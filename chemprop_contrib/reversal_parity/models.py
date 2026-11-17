"""An MPNN whose predictions are exactly parity-equivariant under reaction reversal."""

from __future__ import annotations

import copy

from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.models.model import BatchType
from chemprop.nn import Aggregation, MessagePassing, Predictor
import torch
from torch import Tensor, nn

from chemprop_contrib.reversal_parity.featurizers import ParityCGRFeaturizer, stack_reversed_bmg
from chemprop_contrib.reversal_parity.parity import COMPONENT_POWERS, FREE, build_parity_mask
from chemprop_contrib.reversal_parity.targets import ParityTargetTransform


class ParityEquivariantMPNN(MPNN):
    r"""An :class:`~chemprop.models.MPNN` that is exactly parity-equivariant under reversal.

    With :math:`g` the full stack (message passing, aggregation, batch norm, predictor) and
    :math:`\bar{x}` the reversed reaction, each output head is predicted as

    - odd:  :math:`\tfrac12[g(x) - g(\bar{x})]`, so :math:`f(\bar{x}) = -f(x)`
    - even: :math:`\tfrac12[g(x) + g(\bar{x})]`, so :math:`f(\bar{x}) = f(x)`
    - free: :math:`g(x)`, unconstrained

    A pair of columns (``paired_difference`` or ``paired_reverse``) is predicted from an even
    head and an odd head, and its loss also scores the pair's implied third member. Which
    columns are which is declared by ``target_transform``, which also scales the targets.

    With an uncertainty predictor, the later components of each head (variance, evidential
    parameters, interval width) are combined as even for an odd or even task and left free
    for a free task. For a pair, those of the forward value are free and those of the reverse
    value are the same outputs for :math:`\bar{x}`, so reversal swaps the two; those of the
    difference are even. The three are not forced to be mutually consistent.

    Both passes share every weight, and are stacked on the batch dimension before batch norm
    so that in train mode they see identical batch statistics. The guarantee is exact
    whenever dropout is inactive, as in ``eval()``. With dropout active, the two passes get
    independent masks, so a single draw is only approximately odd or even, although the
    draws for :math:`\bar{x}` are distributed as the negated (odd) or unchanged (even) draws
    for :math:`x`.

    Inputs must be featurized with :class:`ParityCGRFeaturizer`; prefer
    :meth:`from_featurizer`, which reads ``odd_atom_start`` and ``odd_bond_start`` from it.

    Parameters
    ----------
    message_passing, agg, batch_norm, **kwargs
        as for :class:`~chemprop.models.MPNN`
    predictor : Predictor
        ``RegressionFFN``, ``MveFFN``, ``EvidentialFFN`` or ``QuantileFFN``, without an
        ``output_transform`` (the target transform does the unscaling)
    odd_atom_start : int
        first node-feature column of the odd (:math:`d`) block
    odd_bond_start : int
        first edge-feature column of the odd (:math:`d`) block
    target_transform : ParityTargetTransform
        the parity of each target column and the target scaling; the model keeps a copy

    Raises
    ------
    TypeError
        if ``target_transform`` is not a :class:`ParityTargetTransform`.
    ValueError
        if the predictor is not one of the supported regression predictors, has an
        ``output_transform``, or has a different number of tasks than ``target_transform``.

    Notes
    -----
    Inputs must come from :class:`ParityCGRFeaturizer`. Chemprop's own CGR featurizer is not in
    the :math:`(m, d)` basis; :meth:`from_featurizer` rejects it, but the plain constructor only
    sees dimensions and offsets and cannot tell.

    Extra descriptors ``X_d`` and extra atom descriptors ``V_d`` are passed identically to the
    forward and reversed graphs, so they must be direction-invariant.

    Atom/bond feature scaling (``message_passing.graph_transform``) is allowed: it is applied
    inside the encoder, after the reversal, so it is part of :math:`g` and the combination
    stays exact.

    :meth:`fingerprint` returns the forward-direction fingerprint only, so its shape is
    unchanged but it is direction-dependent; the parity structure lives in the combination of
    predictor outputs.
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        odd_atom_start: int,
        odd_bond_start: int,
        target_transform: ParityTargetTransform,
        batch_norm: bool = False,
        **kwargs,
    ):
        # normalized before MPNN.__init__ so that the hyperparameters it collects from this
        # frame hold plain ints
        odd_atom_start = self._as_start_index("odd_atom_start", odd_atom_start)
        odd_bond_start = self._as_start_index("odd_bond_start", odd_bond_start)
        # the upper bounds are checked only when the message passing records its input dims
        # (chemprop's built-in classes do; a custom MessagePassing may not)
        hparams = getattr(message_passing, "hparams", None) or {}
        for name, start, dim_key in (
            ("odd_atom_start", odd_atom_start, "d_v"),
            ("odd_bond_start", odd_bond_start, "d_e"),
        ):
            dim = hparams.get(dim_key)
            if start <= 0 or (dim is not None and start >= dim):
                bound = f"{dim_key} = {dim}" if dim is not None else dim_key
                raise ValueError(f"{name} must satisfy 0 < {name} < {bound}, got {start}")
        if not isinstance(target_transform, ParityTargetTransform):
            raise TypeError(
                "target_transform must be a ParityTargetTransform, got "
                f"{type(target_transform).__name__}"
            )
        # the uncertainty outputs in a pair's forward column are free: the forward pass gives
        # those of the forward value, the reversed pass those of the reverse value
        mask = build_parity_mask(
            target_transform.task_parity, predictor, [p.forward for p in target_transform.pairs]
        )
        if not isinstance(predictor.output_transform, nn.Identity):
            raise ValueError(
                "The predictor has an output_transform. Leave it out: the target transform "
                "scales the targets, after the forward and reversed predictions are combined."
            )

        # The model owns a copy, in train mode like every freshly built module. The transform's
        # mode decides between scaled and unscaled output, so a shared instance (an ensemble
        # built from one transform) or one restored from a checkpoint saved in eval mode would
        # otherwise make training fit unscaled predictions to scaled targets.
        transform = copy.deepcopy(target_transform).train()
        # Kept out of the init arguments Lightning collects from this frame, as chemprop does
        # for X_d_transform: a module there is saved twice and triggers a warning.
        target_transform = None
        super().__init__(message_passing, agg, predictor, batch_norm, **kwargs)
        self.hparams["target_transform"] = transform

        self.odd_atom_start = odd_atom_start
        self.odd_bond_start = odd_bond_start
        self.target_transform = transform
        self._powers = COMPONENT_POWERS[type(predictor)]
        self.register_buffer("parity_mask", mask, persistent=False)
        for metric in (self.criterion, self.metrics[-1]):
            self._weigh_implied_members(metric)

    def _weigh_implied_members(self, metric) -> None:
        """Give the loss a weight for each pair's implied member, after the declared columns.

        An implied reverse value takes its forward column's weight, an implied difference
        weight 1. A single shared weight, the default, needs nothing; a loss restored from a
        checkpoint already has the extra weights.
        """
        pairs = self.target_transform.pairs
        w = metric.task_weights
        if not pairs or w.numel() != self.target_transform.n_tasks or w.numel() == 1:
            return
        extra = [float(w[0, p.forward]) if p.difference is not None else 1.0 for p in pairs]
        extra = torch.tensor([extra], dtype=w.dtype, device=w.device)
        metric.task_weights = torch.cat((w, extra), dim=1)

    @property
    def task_parity(self) -> list[int]:
        """Parity of each output head (``ODD``, ``EVEN`` or ``FREE``, from ``parity``)."""
        return self.target_transform.task_parity

    @staticmethod
    def _as_start_index(name: str, value) -> int:
        try:
            integral = not isinstance(value, bool) and int(value) == value
        except (TypeError, ValueError, OverflowError):  # None, "a", inf
            integral = False
        if not integral:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)

    @classmethod
    def from_featurizer(
        cls,
        featurizer: ParityCGRFeaturizer,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        target_transform: ParityTargetTransform,
        **kwargs,
    ) -> ParityEquivariantMPNN:
        """Build a model whose odd-block offsets are read from ``featurizer``.

        Raises
        ------
        ValueError
            if ``message_passing`` records input widths ``d_v``/``d_e`` that differ from the
            featurizer's ``atom_fdim``/``bond_fdim``.
        TypeError
            if ``featurizer`` is not a :class:`ParityCGRFeaturizer`. Chemprop's own CGR
            featurizers are not in the :math:`(m, d)` basis, so a model built for them would
            silently lose exactness on real reversed reactions.
        """
        if not isinstance(featurizer, ParityCGRFeaturizer):
            raise TypeError(
                "from_featurizer requires a ParityCGRFeaturizer, got "
                f"{type(featurizer).__name__}. Chemprop's CondensedGraphOfReactionFeaturizer is "
                "not in the (m, d) basis that makes reversal a sign flip."
            )
        hparams = getattr(message_passing, "hparams", None) or {}
        for key, fdim in (("d_v", featurizer.atom_fdim), ("d_e", featurizer.bond_fdim)):
            dim = hparams.get(key)
            if dim is not None and dim != fdim:
                raise ValueError(
                    f"message_passing has {key} = {dim} but the featurizer emits {fdim} "
                    "features; build the message passing from featurizer.atom_fdim and "
                    "featurizer.bond_fdim."
                )
        return cls(
            message_passing,
            agg,
            predictor,
            featurizer.odd_atom_start,
            featurizer.odd_bond_start,
            target_transform,
            **kwargs,
        )

    def _paired_fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Fingerprints of the forward and reversed batch, stacked as ``(2B, d)``.

        Both run through the encoder in one pass over a ``2B``-graph batch, which is faster
        than two passes and computes the same thing.
        """
        both = stack_reversed_bmg(bmg, self.odd_atom_start, self.odd_bond_start)
        V_d = None if V_d is None else torch.cat((V_d, V_d))
        H = self.bn(self.agg(self.message_passing(both, V_d), both.batch))

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d).repeat(2, 1)), dim=1)

    def _combine(self, preds: Tensor) -> Tensor:
        """Combine ``(2B, ...)`` stacked predictions into ``(B, ...)`` using the parity mask."""
        B = preds.shape[0] // 2
        fwd, rev = preds[:B], preds[B:]
        m = self.parity_mask
        return torch.where(m == FREE, fwd, 0.5 * (fwd + m * rev))

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate parity-equivariant predictions for the input reactions"""
        both = self.predictor(self._paired_fingerprint(bmg, V_d, X_d))
        return self.target_transform(self._combine(both), self._powers, both[len(both) // 2 :])

    def _train_preds(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Combined ``predictor.train_step`` output, as the training objective sees it: the
        scaled declared columns, followed by each pair's implied member."""
        both = self.predictor.train_step(self._paired_fingerprint(bmg, V_d, X_d))
        return self.target_transform.loss_columns(self._combine(both), both[len(both) // 2 :])

    def on_validation_model_eval(self) -> None:
        # as chemprop does for the predictor's output transform: validation compares scaled
        # predictions with scaled targets
        super().on_validation_model_eval()
        self.target_transform.train()

    def _loss_inputs(self, batch: BatchType) -> tuple[Tensor, ...]:
        """Scaled predictions and targets as the loss sees them, with each pair's implied
        member appended after the declared columns."""
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self._train_preds(bmg, V_d, X_d)
        tf = self.target_transform
        if tf.pairs:
            implied = tf.implied(targets)
            members = [[p.forward, p.second] for p in tf.pairs]
            for m in (lt_mask, gt_mask):
                if m is not None:  # a bound on either member gives no value for the third
                    implied = implied.masked_fill(m[:, members].any(dim=2), float("nan"))
            targets = torch.cat((targets, implied), dim=1)
            none = torch.zeros_like(implied, dtype=torch.bool)
            lt_mask = None if lt_mask is None else torch.cat((lt_mask, none), dim=1)
            gt_mask = None if gt_mask is None else torch.cat((gt_mask, none), dim=1)
        mask = targets.isfinite()
        return preds, targets.nan_to_num(nan=0.0), mask, weights, lt_mask, gt_mask

    # The parent's training/validation steps call `fingerprint` and `predictor.train_step`
    # directly, bypassing `forward`; these mirror them with only the preds computation changed.
    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        l = self.criterion(*self._loss_inputs(batch))

        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)

        return l

    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        self.metrics[-1](*self._loss_inputs(batch))
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True)
