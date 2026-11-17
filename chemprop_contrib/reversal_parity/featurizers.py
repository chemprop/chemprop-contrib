r"""CGR featurizer that emits features in the (mean, difference) basis.

With :math:`x_r` and :math:`x_p` the reactant- and product-side feature vectors, this
featurizer emits :math:`(m, d)` with :math:`m = \tfrac12(x_r + x_p)` and
:math:`d = x_p - x_r`, so that reversing a reaction is a sign flip on the trailing :math:`d`
column block and leaves :math:`m` untouched.

By default (``share_elements=True``) the element is written once, in the even part, as
Chemprop's CGR featurizer does, since a mapped atom keeps its element in a chemical reaction.
With ``share_elements=False`` every atom feature, including the element, gets both parts, so
mapped atoms may change element. This admits alchemical transformations ``A>>B`` (e.g. phenol
to toluene), with the atom map alone defining the topology, as in alchemical free-energy
methods: pairing atoms of different elements is single topology, and sharing only the common
core while leaving the groups that differ unmapped is hybrid topology.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import Iterable

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph.reaction import (
    CondensedGraphOfReactionFeaturizer,
    RxnMode,
)
import numpy as np
from rdkit.Chem.rdchem import Bond, Mol
import torch


@dataclass
class ParityCGRFeaturizer(CondensedGraphOfReactionFeaturizer):
    r"""Featurize reactions as a CGR in the :math:`(m, d)` basis.

    Parameters
    ----------
    balanced : bool, default=False
        whether to use the ``*_BALANCE`` bond behavior of the parent featurizer
        (``REAC_DIFF_BALANCE`` instead of ``REAC_DIFF``)
    share_elements : bool, default=True
        whether to write the element once, in the even block only, as Chemprop's
        ``REAC_DIFF`` does. Mapped atoms must then keep their element, and featurizing a
        reaction in which one does not raises :class:`ValueError`. Pass ``False`` for
        alchemical transformations, in which mapped atoms may change element.
    mode_ : None
        not configurable; this featurizer defines its own layout. Passing a value raises
        :class:`ValueError`.

    Notes
    -----
    Node layout (``atom_fdim = 2F``): ``[0:F]`` is the even block :math:`m`, ``[F:2F]`` the
    odd block :math:`d`. With ``share_elements=True``, :math:`d` omits the element one-hot of
    width ``k = len(atom_featurizer.atomic_nums) + 1``, so ``atom_fdim = 2F - k``, the same as
    Chemprop's ``REAC_DIFF``, and the odd block is ``[F:2F-k]``. Edge layout
    (``bond_fdim = 2B``): ``[0:B]`` is the even block :math:`m`, ``[B:2B]`` the odd block
    :math:`d`.

    An atom absent from one side is a dummy atom there. With ``share_elements=True`` the
    dummy keeps the element (``atom_featurizer.num_only``), as in Chemprop; with
    ``share_elements=False`` it is a zero vector (``atom_featurizer(None)``), so a vanishing
    atom has :math:`d = -x_r`. With ``balanced=True`` an absent
    atom is instead copied across unchanged (:math:`d = 0`), Chemprop's ``*_BALANCE`` meaning
    of an unrecorded spectator. With both options, the features are Chemprop's ``REAC_DIFF``
    (or ``REAC_DIFF_BALANCE``) features :math:`(x_r, d)` mapped to :math:`(x_r + d/2, d)`.
    """

    # keyword-only, so that a third positional argument cannot land in ``mode_``
    _: KW_ONLY
    balanced: bool = False
    share_elements: bool = True
    mode_: InitVar[str | RxnMode | None] = None

    def __post_init__(self, mode_: str | RxnMode | None):
        if mode_ is not None:
            raise ValueError(
                "ParityCGRFeaturizer defines its own feature layout and does not accept "
                f"'mode_' (got {mode_!r}); use 'balanced' to select the balance behavior."
            )
        super().__post_init__(RxnMode.REAC_DIFF_BALANCE if self.balanced else RxnMode.REAC_DIFF)
        self.atom_fdim = 2 * len(self.atom_featurizer)
        if self.share_elements:
            self.atom_fdim -= self._n_element_cols

    @property
    def _n_element_cols(self) -> int:
        # the element one-hot leads the atom features: one bit per listed element plus "other"
        return len(self.atom_featurizer.atomic_nums) + 1

    @property
    def odd_atom_start(self) -> int:
        """First column of the odd (:math:`d`) block of the node features."""
        return len(self.atom_featurizer)

    @property
    def odd_bond_start(self) -> int:
        """First column of the odd (:math:`d`) block of the edge features."""
        return len(self.bond_featurizer)

    def _absent_side(self, x: np.ndarray, atom) -> np.ndarray:
        """Features, on the side where it is absent, of an unmapped atom featurized as ``x``:
        a copy (``balanced``), its element only (``share_elements``), or a zero dummy atom."""
        if self.balanced:
            return x
        if self.share_elements:
            return self.atom_featurizer.num_only(atom)
        return self.atom_featurizer(None)

    def _calc_node_feature_matrix(
        self,
        rct: Mol,
        pdt: Mol,
        r2p_idx_map: dict[int, int],
        pdt_idxs: Iterable[int],
        reac_idxs: Iterable[int],
    ) -> np.ndarray:
        # Built here rather than from the parent's output, which writes the element once
        # (from whichever side is present) and so cannot represent an element change.
        f = self.atom_featurizer
        reac_only = set(reac_idxs)
        X_r, X_p = [], []
        for a in rct.GetAtoms():
            x_r = f(a)
            if a.GetIdx() in reac_only:
                x_p = self._absent_side(x_r, a)
            else:
                b = pdt.GetAtomWithIdx(r2p_idx_map[a.GetIdx()])
                if self.share_elements and a.GetAtomicNum() != b.GetAtomicNum():
                    raise ValueError(
                        f"mapped atom {a.GetAtomMapNum()} changes element ({a.GetSymbol()} -> "
                        f"{b.GetSymbol()}), which share_elements=True cannot represent; use "
                        "share_elements=False for alchemical transformations."
                    )
                x_p = f(b)
            X_r.append(x_r)
            X_p.append(x_p)
        for j in pdt_idxs:
            b = pdt.GetAtomWithIdx(j)
            x_p = f(b)
            X_r.append(self._absent_side(x_p, b))
            X_p.append(x_p)

        X_r = np.array(X_r, dtype=float).reshape(-1, len(f))
        X_p = np.array(X_p, dtype=float).reshape(-1, len(f))
        D = X_p - X_r
        if self.share_elements:
            D = D[:, self._n_element_cols :]
        return np.hstack((0.5 * (X_r + X_p), D))

    def _calc_edge_feature(self, b_reac: Bond | None, b_pdt: Bond | None) -> np.ndarray:
        # the parent's bond features are integer-valued; the half-sum needs a float dtype
        x_e = super()._calc_edge_feature(b_reac, b_pdt).astype(float)
        Bb = self.odd_bond_start
        x_e[:Bb] += 0.5 * x_e[Bb:]
        return x_e


def reverse_bmg(bmg: BatchMolGraph, odd_atom_start: int, odd_bond_start: int) -> BatchMolGraph:
    r"""Reverse every reaction in a batch by negating the odd feature blocks.

    In the :math:`(m, d)` basis reversal is :math:`d \to -d` with :math:`m` unchanged, and the
    graph topology is identical, so ``edge_index``, ``rev_edge_index`` and ``batch`` are
    shared with the input. The input is not modified.

    Parameters
    ----------
    bmg : BatchMolGraph
        batch featurized with :class:`ParityCGRFeaturizer`
    odd_atom_start : int
        first node-feature column of the odd block
    odd_bond_start : int
        first edge-feature column of the odd block
    """
    V = bmg.V.clone()
    E = bmg.E.clone()
    V[:, odd_atom_start:] *= -1
    E[:, odd_bond_start:] *= -1

    out = copy.copy(bmg)
    out.V = V
    out.E = E
    return out


# the BatchMolGraph fields stack_reversed_bmg knows how to stack; the size is name-mangled
_STACKED_FIELDS = {"V", "E", "edge_index", "rev_edge_index", "batch", "_BatchMolGraph__size"}


def stack_reversed_bmg(
    bmg: BatchMolGraph, odd_atom_start: int, odd_bond_start: int
) -> BatchMolGraph:
    """Stack a batch and its reversal into one batch of ``2B`` graphs.

    Graphs ``0..B-1`` are the input and graphs ``B..2B-1`` their reversals, in the same order,
    so one encoder pass computes what :func:`reverse_bmg` and two passes would. The result
    equals chemprop's collate of the forward graphs followed by the reversed ones. The input
    is not modified.

    Parameters
    ----------
    bmg : BatchMolGraph
        batch featurized with :class:`ParityCGRFeaturizer`
    odd_atom_start : int
        first node-feature column of the odd block
    odd_bond_start : int
        first edge-feature column of the odd block

    Raises
    ------
    TypeError
        if the batch carries fields or attributes other than chemprop's ``BatchMolGraph``
        ones, which this function would otherwise silently leave out of the result
    """
    names = {f.name for f in dataclasses.fields(bmg)} | set(getattr(bmg, "__dict__", {}))
    if unknown := sorted(names - _STACKED_FIELDS):
        raise TypeError(
            f"cannot stack a {type(bmg).__name__} with fields {unknown}; only the fields of "
            "chemprop's BatchMolGraph (V, E, edge_index, rev_edge_index, batch) are supported."
        )
    rev = reverse_bmg(bmg, odd_atom_start, odd_bond_start)
    n_atoms, n_edges = bmg.V.shape[0], bmg.E.shape[0]

    out = copy.copy(bmg)
    out.V = torch.cat((bmg.V, rev.V))
    out.E = torch.cat((bmg.E, rev.E))
    out.edge_index = torch.cat((bmg.edge_index, bmg.edge_index + n_atoms), dim=1)
    out.rev_edge_index = torch.cat((bmg.rev_edge_index, bmg.rev_edge_index + n_edges))
    out.batch = torch.cat((bmg.batch, bmg.batch + len(bmg)))
    out._BatchMolGraph__size = 2 * len(bmg)
    return out
