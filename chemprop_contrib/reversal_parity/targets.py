"""Target parity and scaling for parity-equivariant reaction models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import operator

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn

from chemprop_contrib.reversal_parity.parity import EVEN, FREE, ODD


def _index(name: str, value, n_tasks: int) -> int:
    try:
        i = None if isinstance(value, bool) else operator.index(value)
    except TypeError:
        i = None
    if i is None or not 0 <= i < n_tasks:
        raise ValueError(f"{name} must hold column indices in [0, {n_tasks}), got {value!r}")
    return i


def _indices(name: str, values: Iterable, n_tasks: int) -> list[int]:
    if isinstance(values, (Mapping, str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be a list of column indices, got {values!r}")
    out = [_index(name, v, n_tasks) for v in values]
    if len(set(out)) != len(out):
        raise ValueError(f"{name} lists a column more than once: {out}")
    return sorted(out)


@dataclass(frozen=True)
class _Pair:
    """A forward column and either its reverse or its difference column (the other is None)."""

    forward: int
    reverse: int | None = None
    difference: int | None = None

    @property
    def second(self) -> int:
        """The column given besides ``forward``."""
        return self.reverse if self.reverse is not None else self.difference

    def __str__(self) -> str:
        kind = "reverse" if self.reverse is not None else "difference"
        return f"paired_{kind}={{{self.forward}: {self.second}}}"


class ParityTargetTransform(nn.Module):
    r"""Declares how each target column behaves under reversal, and scales the targets.

    Every column not listed is free (unconstrained).

    - ``odd``: :math:`f(\bar{x}) = -f(x)`, e.g. :math:`\Delta H`, :math:`\Delta\Delta G`,
      :math:`\log K_{\text{eq}}`
    - ``even``: :math:`f(\bar{x}) = f(x)`
    - ``paired_reverse``: ``{forward: reverse}``, a property given in both directions, which
      reversal swaps: :math:`f_{\text{rev}}(x) = f_{\text{fwd}}(\bar{x})`, as for the forward and
      reverse barriers or the forward and reverse rate constants (:math:`\log k`) of a reaction
    - ``paired_difference``: ``{forward: difference}``, the same kind of property given by its
      forward value and its difference :math:`f_{\text{diff}} = f_{\text{fwd}} - f_{\text{rev}}`
      (for barriers, the reaction energy; for :math:`\log k`, :math:`\log K_{\text{eq}}`)

    The model predicts a pair from an even mean :math:`M` and an odd difference :math:`D`, as
    :math:`f_{\text{fwd}} = M + \tfrac12 D`, :math:`f_{\text{rev}} = M - \tfrac12 D` and
    :math:`f_{\text{diff}} = D`. The loss scores all three members of a pair: a row whose two
    labels are known also scores the implied third, and a row with one label scores that one,
    so both declarations train the same model on the same data. The model outputs only the
    columns of the dataset.

    With an uncertainty predictor, each member of a pair has its own uncertainty components
    :math:`u`: :math:`u_{\text{rev}}(x) = u_{\text{fwd}}(\bar{x})`, with :math:`u_{\text{fwd}}`
    unconstrained, and :math:`u_{\text{diff}}` is even.

    The model works in the scaled target space and applies this transform after combining the
    forward and reversed predictions. In train mode it returns scaled predictions, which the
    loss compares with targets normalized by :meth:`to_standard_scaler`; in eval mode it also
    unscales them. Odd columns and differences are scaled without a shift, since a shift is
    even and would break their parity, and the two directions of a pair share one scaling.

    Build it with :meth:`fit` from the training targets, and normalize the datasets with
    :meth:`to_standard_scaler`.

    Parameters
    ----------
    n_tasks : int
        number of target columns
    odd, even : Iterable[int]
        indices of the odd and even columns
    paired_difference : Mapping[int, int] | None
        ``{forward column: difference column}``
    paired_reverse : Mapping[int, int] | None
        ``{forward column: reverse column}``
    mean, scale : ArrayLike | None
        per-column target mean and scale; default 0 and 1 (no scaling). Odd and difference
        columns must have mean 0, and the forward and reverse columns of a pair the same mean
        and scale.
    difference_scale : ArrayLike | None
        scale of each pair's difference, one per pair in increasing order of the forward
        column. For a pair in ``paired_difference`` it is the difference column's scale; for
        one in ``paired_reverse`` it defaults to the scale of the forward column.

    Raises
    ------
    ValueError
        if an index is out of range, a column is given two roles, a scaling is inconsistent
        with the declaration, or a scale is not positive.
    """

    def __init__(
        self,
        n_tasks: int,
        odd: Iterable[int] = (),
        even: Iterable[int] = (),
        paired_difference: Mapping[int, int] | None = None,
        paired_reverse: Mapping[int, int] | None = None,
        mean: ArrayLike | None = None,
        scale: ArrayLike | None = None,
        difference_scale: ArrayLike | None = None,
    ):
        super().__init__()
        try:
            n = None if isinstance(n_tasks, bool) else operator.index(n_tasks)
        except TypeError:
            n = None
        if n is None or n < 1:
            raise ValueError(f"n_tasks must be a positive integer, got {n_tasks!r}")
        n_tasks = n
        odd, even = _indices("odd", odd, n_tasks), _indices("even", even, n_tasks)
        paired = self._parse_pairs(paired_difference, paired_reverse, n_tasks)
        roles = [*odd, *even, *(i for p in paired for i in (p.forward, p.second))]
        if twice := sorted({i for i in roles if roles.count(i) > 1}):
            raise ValueError(f"column(s) {twice} are given more than one role")

        self.n_tasks = n_tasks
        self.odd = odd
        self.even = even
        self.paired_difference = {
            p.forward: p.difference for p in paired if p.difference is not None
        }
        self.paired_reverse = {p.forward: p.reverse for p in paired if p.reverse is not None}
        self.pairs = paired  # both kinds, in increasing order of the forward column

        mean = np.zeros(n_tasks) if mean is None else np.asarray(mean, dtype=float)
        scale = np.ones(n_tasks) if scale is None else np.asarray(scale, dtype=float)
        for name, a in (("mean", mean), ("scale", scale)):
            if a.shape != (n_tasks,) or not np.isfinite(a).all():
                raise ValueError(f"{name} must hold {n_tasks} finite values, got {a.tolist()}")
        if (scale <= 0).any():
            raise ValueError(f"scale must be positive, got {scale.tolist()}")
        zero_mean = [*odd, *(p.difference for p in paired if p.difference is not None)]
        if bad := sorted(i for i in zero_mean if mean[i] != 0):
            raise ValueError(
                f"odd or difference column(s) {bad} have a nonzero mean; a shift is even and "
                "would break their parity. Use ParityTargetTransform.fit, which scales them "
                "about zero."
            )
        for p in paired:
            r = p.reverse
            if r is not None and (mean[r] != mean[p.forward] or scale[r] != scale[p.forward]):
                raise ValueError(f"{p}: forward and reverse columns must share mean and scale")

        dscale = np.array(
            [scale[p.difference if p.difference is not None else p.forward] for p in paired]
        )
        if difference_scale is not None:
            given = np.asarray(difference_scale, dtype=float)
            if given.shape != (len(paired),) or not np.isfinite(given).all() or (given <= 0).any():
                raise ValueError(
                    f"difference_scale must hold {len(paired)} positive values, got "
                    f"{given.tolist()}"
                )
            mismatched = zip(paired, given, dscale)
            if bad := [str(p) for p, g, s in mismatched if p.difference is not None and g != s]:
                raise ValueError(
                    f"difference_scale must equal the difference column's scale for {bad}"
                )
            dscale = given

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float).unsqueeze(0))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float).unsqueeze(0))
        self.register_buffer("difference_scale", torch.tensor(dscale, dtype=torch.float)[None])
        # The pairs' columns, as index tensors that follow the module's device. They restate
        # the declaration, so they stay out of the state dict.
        for name, values, dtype in (
            ("_fwd", [p.forward for p in paired], torch.long),
            ("_sec", [p.second for p in paired], torch.long),
            ("_is_rev", [p.reverse is not None for p in paired], torch.bool),
        ):
            self.register_buffer(name, torch.tensor(values, dtype=dtype), persistent=False)

    @staticmethod
    def _parse_pairs(paired_difference, paired_reverse, n_tasks: int) -> list[_Pair]:
        out = []
        for name, kind, mapping in (
            ("paired_difference", "difference", paired_difference),
            ("paired_reverse", "reverse", paired_reverse),
        ):
            if mapping is None:
                continue
            if not isinstance(mapping, Mapping):
                raise ValueError(
                    f"{name} must map forward columns to {kind} columns, e.g. {name}={{0: 1}}, "
                    f"got {mapping!r}"
                )
            for k, v in mapping.items():
                pair = _Pair(_index(name, k, n_tasks), **{kind: _index(name, v, n_tasks)})
                if pair.forward == pair.second:
                    raise ValueError(f"{name} pairs column {pair.forward} with itself")
                out.append(pair)
        return sorted(out, key=lambda p: p.forward)

    @classmethod
    def fit(
        cls,
        Y: ArrayLike,
        odd: Iterable[int] = (),
        even: Iterable[int] = (),
        paired_difference: Mapping[int, int] | None = None,
        paired_reverse: Mapping[int, int] | None = None,
    ) -> ParityTargetTransform:
        """Fit the scaling to training targets ``Y`` of shape ``(n_samples, n_tasks)``.

        Odd columns get mean 0 and their root mean square as the scale, since their standard
        deviation about a nonzero mean can misjudge their size badly when the data mostly run
        in one direction. The two directions of a pair share the mean and standard deviation
        of their values pooled, with the missing direction implied by the difference where the
        row has both labels, and the difference gets mean 0 and its root mean square. Other
        columns get their mean and standard deviation. Missing values (NaN) are ignored.

        Raises
        ------
        ValueError
            if ``Y`` is not 2-D, a column has no finite values, a pair in ``paired_reverse``
            has no row with both labels, or the parity declaration is invalid.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n_samples, n_tasks), got shape {Y.shape}")
        spec = cls(Y.shape[1], odd, even, paired_difference, paired_reverse)
        if inf := np.flatnonzero(np.isinf(Y).any(axis=0)).tolist():
            raise ValueError(f"column(s) {inf} hold infinite target values")
        if empty := np.flatnonzero(~np.isfinite(Y).any(axis=0)).tolist():
            raise ValueError(f"column(s) {empty} have no finite target values")

        def rms(a):
            return np.sqrt(np.nanmean(a**2, axis=0))

        mean, scale = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
        mean[spec.odd] = 0.0
        scale[spec.odd] = rms(Y[:, spec.odd])
        dscale = []
        for p in spec.pairs:
            F, S = Y[:, p.forward], Y[:, p.second]
            R, D = (S, F - S) if p.reverse is not None else (F - S, S)
            if not np.isfinite(D).any():
                raise ValueError(f"{p}: no row has both labels, so the difference has no scale")
            both = np.concatenate([F, R])
            mean[p.forward], scale[p.forward] = np.nanmean(both), np.nanstd(both)
            dscale.append(rms(D))
            if p.reverse is not None:
                mean[p.reverse], scale[p.reverse] = mean[p.forward], scale[p.forward]
            else:
                mean[p.difference], scale[p.difference] = 0.0, dscale[-1]
        scale[scale == 0] = 1.0  # as StandardScaler does for constant columns
        dscale = np.array(dscale, dtype=float)
        dscale[dscale == 0] = 1.0
        return cls(
            Y.shape[1],
            spec.odd,
            spec.even,
            spec.paired_difference,
            spec.paired_reverse,
            mean=mean,
            scale=scale,
            difference_scale=dscale,
        )

    def to_standard_scaler(self) -> StandardScaler:
        """A fitted :class:`~sklearn.preprocessing.StandardScaler` with this scaling, for
        ``dataset.normalize_targets``."""
        scaler = StandardScaler()
        scaler.mean_ = self.mean[0].detach().double().cpu().numpy()
        scaler.scale_ = self.scale[0].detach().double().cpu().numpy()
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = self.n_tasks
        return scaler

    @property
    def task_parity(self) -> list[int]:
        """Parity of each model output head.

        A pair is predicted by an even head (its mean) in the forward column and an odd head
        (its difference) in the other column.
        """
        parity = [FREE] * self.n_tasks
        for i in self.odd:
            parity[i] = ODD
        for i in self.even:
            parity[i] = EVEN
        for p in self.pairs:
            parity[p.forward], parity[p.second] = EVEN, ODD
        return parity

    def _columns(self, heads: Tensor, reverse: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """The scaled declared columns, and the uncertainty components of each pair's implied
        member (``None`` for 2-D ``heads``)."""
        if not self.pairs:
            return heads, None
        if heads.ndim == 3 and reverse is None:
            raise ValueError(
                "Pairs with uncertainty components need the reversed pass: the forward column's "
                "components there are the uncertainty of the pair's reverse value."
            )
        # heads: the forward column holds the scaled mean M, the other one the difference D
        # in units of difference_scale. Scaled, forward = M + c D, reverse = M - c D.
        fwd, sec, is_rev = self._fwd, self._sec, self._is_rev
        # half the difference, in units of the forward column's scale
        c = (self.difference_scale / (2 * self.scale[:, fwd])).to(heads.dtype)
        cols = heads.clone()
        value = cols if heads.ndim == 2 else cols[..., 0]  # a view of cols
        M, D = value[:, fwd], value[:, sec]  # copies, as the indices are tensors
        value[:, fwd] = M + c * D
        value[:, sec] = torch.where(is_rev, M - c * D, D)
        if heads.ndim == 2:
            return cols, None
        # The forward value keeps its own uncertainty components, which are free; those of the
        # reverse value are the same outputs for the reversed input, and those of the
        # difference are the even ones in the pair's other column.
        u_rev, u_diff = reverse[:, fwd, 1:], heads[:, sec, 1:]
        k = is_rev[:, None]
        cols[:, sec, 1:] = torch.where(k, u_rev, u_diff)
        return cols, torch.where(k, u_diff, u_rev)

    def forward(
        self, preds: Tensor, powers: Sequence[int] = (), reverse: Tensor | None = None
    ) -> Tensor:
        """Map model heads to the declared columns, then (in eval mode) unscale.

        ``preds`` is ``(B, n_tasks)``, or ``(B, n_tasks, n_targets)`` for uncertainty
        predictors, whose component 0 is the prediction itself and component ``j > 0`` is
        multiplied by ``scale ** powers[j]`` on unscaling (2 for a variance, 1 for an interval
        width, 0 for a dimensionless parameter). Pairs with uncertainty components also need
        ``reverse``, the predictor's output for the reversed inputs, in the same shape.
        """
        preds, _ = self._columns(preds, reverse)
        if self.training:
            return preds
        if preds.ndim == 2:
            return preds * self.scale + self.mean
        out = preds.clone()
        out[..., 0] = preds[..., 0] * self.scale + self.mean
        for j in range(1, preds.shape[-1]):
            out[..., j] = preds[..., j] * self.scale ** powers[j]
        return out

    def loss_columns(self, heads: Tensor, reverse: Tensor | None = None) -> Tensor:
        """The scaled declared columns followed by each pair's implied member, as the loss
        scores them: ``(B, n_tasks + n_pairs)``, with the components of ``heads`` if it is 3-D.

        ``heads`` and ``reverse`` are as for :meth:`forward`.
        """
        cols, uncertainty = self._columns(heads, reverse)
        if not self.pairs:
            return cols
        if uncertainty is None:
            return torch.cat((cols, self.implied(cols)), dim=1)
        member = torch.cat((self.implied(cols[..., 0]).unsqueeze(-1), uncertainty), dim=-1)
        return torch.cat((cols, member), dim=1)

    def implied(self, Z: Tensor) -> Tensor:
        """The implied third member of each pair, ``(B, n_pairs)``, from scaled columns ``Z``.

        ``Z`` holds scaled values of the declared columns, ``(B, n_tasks)``: predictions in
        train mode, or normalized targets. The implied member, forward minus the other column,
        is the difference for a pair in ``paired_reverse`` and the reverse value for one in
        ``paired_difference``; it is returned scaled like that member (so a missing label gives
        NaN).
        """
        fwd, sec, is_rev = self._fwd, self._sec, self._is_rev
        mean, scale = self.mean.to(Z.dtype), self.scale.to(Z.dtype)
        F = Z[:, fwd] * scale[:, fwd] + mean[:, fwd]
        S = Z[:, sec] * scale[:, sec] + mean[:, sec]
        dscale = self.difference_scale.to(Z.dtype)
        member_mean = torch.where(is_rev, torch.zeros_like(F[:1]), mean[:, fwd])
        member_scale = torch.where(is_rev, dscale, scale[:, fwd])
        return (F - S - member_mean) / member_scale

    def extra_repr(self) -> str:
        return (
            f"n_tasks={self.n_tasks}, odd={self.odd}, even={self.even}, "
            f"paired_difference={self.paired_difference}, paired_reverse={self.paired_reverse}"
        )
