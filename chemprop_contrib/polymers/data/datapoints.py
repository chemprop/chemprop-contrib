from __future__ import annotations

from dataclasses import dataclass, field

from rdkit.Chem import AllChem as Chem

from chemprop.data.datapoints import MoleculeDatapoint
from chemprop_contrib.polymers.utils import make_polymer_mol


@dataclass
class _PolymerDatapointMixin:
    mol: Chem.Mol
    """the polymer associated with this datapoint"""

    @classmethod
    def from_smi(
        cls,
        smi: str,
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        **kwargs,
    ) -> _PolymerDatapointMixin:
        frag_weights = smi.split("|")[1:-1]
        edge_rules = smi.split("<")[1:]
        mol = make_polymer_mol(smi.split("|")[0], keep_h, add_h, ignore_stereo, reorder_atoms)

        kwargs["name"] = smi if "name" not in kwargs else kwargs["name"]

        return cls(mol, fragment_weights=frag_weights, edge_rules=edge_rules, *args, **kwargs)


# NOTE: the original PR built this class from mixins introduced by its refactor of chemprop's
# datapoints.py. Subclassing MoleculeDatapoint provides the same V_f/E_f/V_d fields and NaN
# handling, and lets chemprop's data splitting accept polymer datapoints.
@dataclass(kw_only=True)
class PolymerDatapoint(_PolymerDatapointMixin, MoleculeDatapoint):
    """A :class:`PolymerDatapoint` contains a single polymer and its associated features and targets."""

    fragment_weights: list[str] = field(default_factory=list)
    """the fragment weights associated with each monomer"""
    edge_rules: list[str] | None = None
    """the directed edges between monomers"""

    def __post_init__(self):
        super(PolymerDatapoint, self).__post_init__()

    @property
    def polymer(self) -> tuple[Chem.Mol, list[str], list[str]]:
        return (self.mol, self.fragment_weights, self.edge_rules)
