"""Per-task parity masks for parity-equivariant reaction models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import operator

import torch
from torch import Tensor

from chemprop.nn.predictors import EvidentialFFN, MveFFN, Predictor, QuantileFFN, RegressionFFN

ODD = -1
EVEN = 1
FREE = 0

_VALID_PARITIES = (ODD, EVEN, FREE)

# Exact types only: MveFFN, EvidentialFFN and QuantileFFN subclass RegressionFFN, and any other
# subclass (e.g. SpectralFFN, whose output is a normalized positive spectrum) must be rejected
# rather than admitted silently.
_ALLOWED_PREDICTORS = frozenset({RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN})

# How each output component of a predictor unscales, as in chemprop's own output transform:
# component 0 (the prediction) is affine; component j > 0 is multiplied by scale ** power[j]
# (variance 2, interval width 1, dimensionless evidential parameters 0).
COMPONENT_POWERS = {
    RegressionFFN: (1,),
    MveFFN: (1, 2),
    EvidentialFFN: (1, 0, 0, 2),
    QuantileFFN: (1, 1),
}


def normalize_task_parity(task_parity: Sequence[int]) -> list[int]:
    """Validate ``task_parity`` and return it as a list of plain ``int``.

    Integer-like values (``numpy`` or ``torch`` integer scalars) are accepted; floats, booleans
    and any value outside ``ODD``, ``EVEN``, ``FREE`` raise ``ValueError``, so that e.g. ``0.7``
    is never truncated to ``FREE``.
    """
    out, bad = [], []
    for p in task_parity:
        try:
            value = None if isinstance(p, bool) else operator.index(p)
        except TypeError:
            value = None
        if value not in _VALID_PARITIES:
            bad.append(p)
        else:
            out.append(value)
    if bad:
        raise ValueError(f"Invalid parity values {bad}; each must be one of ODD, EVEN, FREE.")
    return out


def build_parity_mask(
    task_parity: Sequence[int], predictor: Predictor, free_uncertainty: Iterable[int] = ()
) -> Tensor:
    r"""Build the mask applied to predictor outputs when combining forward and reversed passes.

    Component 0 of each task takes that task's parity. The later components (variance,
    evidential parameters, interval width) describe its uncertainty:

    - odd or even task: ``EVEN``. Reversal maps the value to itself up to sign, so its
      uncertainty is unchanged. For a quantile interval this is the swap of an odd task's
      bounds, :math:`L(\bar{x}) = -U(x)`.
    - free task: ``FREE``, as nothing relates its values for :math:`x` and :math:`\bar{x}`.

    Parameters
    ----------
    task_parity : Sequence[int]
        parity of each task (``ODD``, ``EVEN`` or ``FREE``)
    predictor : Predictor
        the predictor whose outputs are combined
    free_uncertainty : Iterable[int]
        tasks whose later components are ``FREE`` whatever their parity, as for the forward
        column of a pair: its two passes give the uncertainties of the pair's two directions

    Returns
    -------
    Tensor
        Float tensor of shape ``(n_tasks,)`` if ``predictor.n_targets == 1``, else
        ``(n_tasks, n_targets)``.
    """
    if type(predictor) not in _ALLOWED_PREDICTORS:
        allowed = ", ".join(sorted(c.__name__ for c in _ALLOWED_PREDICTORS))
        raise ValueError(
            f"Parity masks are only defined for {allowed}; got {type(predictor).__name__}."
        )
    task_parity = normalize_task_parity(task_parity)
    if len(task_parity) != predictor.n_tasks:
        raise ValueError(
            f"task_parity has length {len(task_parity)} but predictor has "
            f"n_tasks={predictor.n_tasks}."
        )

    mask = torch.tensor(task_parity, dtype=torch.get_default_dtype())
    if predictor.n_targets == 1:
        return mask
    full = torch.ones(predictor.n_tasks, predictor.n_targets, dtype=mask.dtype)
    full[:, 0] = mask
    full[mask == FREE, 1:] = FREE
    full[list(free_uncertainty), 1:] = FREE
    return full
