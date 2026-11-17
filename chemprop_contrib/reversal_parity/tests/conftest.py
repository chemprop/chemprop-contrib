"""Shared helpers and fixtures for the reversal_parity tests."""

from __future__ import annotations

from chemprop.data.collate import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.utils import make_mol
from rdkit import Chem
import torch
from torch import Tensor

# Copied from chemprop tests/unit/featurizers/test_cgr.py:51-59 at commit 1b90e3e2.
# Covers balanced+mapped, reordered atom indices, partially mapped on each side,
# product-heavy, reactant-heavy, and a zero-edge reaction.
rxn_smis = [
    # reactant and product with the same number of atoms
    "[CH3:1][H:2]>>[CH3:1].[H:2]",  # reactant and product are balanced and mapped
    "[CH3:2][H:1]>>[H:1].[CH3:2]",  # balanced, mapped but with different atom index order
    "[CH3:1][H]>>[CH3:1].[H:2]",  # balanced, but reactant has less atom-mapped atoms
    "[CH3:1][H:2]>>[H].[CH3:1]",  # balanced, but product has less atom-mapped atoms
    # reactant and product has different numbers of atoms
    "[CH4:1]>>[CH2:1].[H:2][H:3]",  # product has more atoms and more atom-mapped atoms
    "[H:1].[CH2:2][H:3]>>[CH3:2][H:3]",  # reactant with more atoms and atom-mapped atoms
    "[CH4:1]>>[CH3:1].[H:2]",  # product with more atoms and atom-mapped atoms with 0 edge
]


_RING = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1"

# Alchemical transformations: mapped atoms may change element. The atom map alone encodes the
# topology; ordinary reactions are the element-preserving special case.
alchemical_smis = [
    f"{_RING}[OH:7]>>{_RING}[CH3:7]",  # phenol -> toluene, single topology (O -> C in place)
    # phenol -> toluene, hybrid topology (shared ring; O vanishes, C appears)
    f"{_RING}[OH]>>{_RING}[CH3]",
    # single topology with dummy atoms: O -> C mapped, hydroxyl H mapped to one methyl H, two
    # methyl H appear
    f"{_RING}[O:7][H:8]>>{_RING}[C:7]([H:8])([H])[H]",
    "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][n:6]1",  # in-ring
]


# every input the parity guarantee must hold for: reactions and alchemical pairs
parity_smis = rxn_smis + alchemical_smis

# the inputs that share_elements=True accepts: no mapped atom changes element (atoms that
# vanish or appear unmapped, as in hybrid topology, are allowed)
shared_element_smis = rxn_smis + alchemical_smis[1:2]


def make_rxn(smi: str) -> tuple[Chem.Mol, Chem.Mol]:
    """Split a reaction SMILES on ``>>`` into the ``(reactant, product)`` pair that a
    ``CondensedGraphOfReactionFeaturizer`` expects."""
    rct, pdt = smi.split(">>")
    return (
        make_mol(rct, keep_h=True, add_h=False),
        make_mol(pdt, keep_h=True, add_h=False),
    )


def reverse_rxn_smi(smi: str) -> str:
    rct, pdt = smi.split(">>")
    return f"{pdt}>>{rct}"


_ENCODERS: dict[tuple[int, int], tuple[BondMessagePassing, MeanAggregation]] = {}


def _get_encoder(d_v: int, d_e: int):
    # cached so that every call with the same dims uses identical weights
    if (d_v, d_e) not in _ENCODERS:
        torch.manual_seed(0)
        mp = BondMessagePassing(d_v=d_v, d_e=d_e, d_h=32).eval()
        _ENCODERS[(d_v, d_e)] = (mp, MeanAggregation().eval())
    return _ENCODERS[(d_v, d_e)]


@torch.no_grad()
def encode(mg: MolGraph | BatchMolGraph) -> Tensor:
    """Graph-level fingerprint from a seeded, untrained ``BondMessagePassing`` + mean
    aggregation. Returns a ``(n_graphs, d)`` tensor; ``(1, d)`` for a single ``MolGraph``."""
    bmg = BatchMolGraph([mg]) if isinstance(mg, MolGraph) else mg
    mp, agg = _get_encoder(bmg.V.shape[1], bmg.E.shape[1])
    return agg(mp(bmg), bmg.batch)
