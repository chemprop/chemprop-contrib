r"""Polymer versions of :func:`chemprop.cli.utils.parsing.build_data_from_files` and
:func:`chemprop.cli.utils.parsing.make_dataset`.

``chemprop-polymer`` stores the ``--polymer-columns`` in ``args.smiles_columns``, so chemprop's
:func:`~chemprop.cli.utils.parsing.parse_csv` returns the polymer strings as ``smiss``. The
polymer branches below are lifted from the original PR's edits to chemprop's ``make_datapoints``
and ``make_dataset``.
"""

import logging
from os import PathLike
from typing import Literal, Mapping, Sequence

import numpy as np
from rdkit.Chem.rdchem import RWMol

from chemprop.cli.utils import parsing
from chemprop.cli.utils.parsing import load_input_feats_and_descs, parse_csv
from chemprop.featurizers.atom import get_multi_hot_atom_featurizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer, RIGRBondFeaturizer
from chemprop.featurizers.molecule import MoleculeFeaturizerRegistry
from chemprop.featurizers.molgraph import RxnMode
from chemprop_contrib.polymers.data import PolymerDatapoint, PolymerDataset
from chemprop_contrib.polymers.featurizers import PolymerMolGraphFeaturizer
from chemprop_contrib.polymers.utils import make_polymer_mol, remove_wildcard_atoms

logger = logging.getLogger(__name__)


def make_polymer_datapoints(
    polyss: list[list[str]],
    Y: np.ndarray,
    weights: np.ndarray | None,
    lt_mask: np.ndarray | None,
    gt_mask: np.ndarray | None,
    X_d: np.ndarray | None,
    V_fss: list[list[np.ndarray] | list[None]] | None,
    E_fss: list[list[np.ndarray] | list[None]] | None,
    V_dss: list[list[np.ndarray] | list[None]] | None,
    molecule_featurizers: list[str] | None,
    keep_h: bool,
    add_h: bool,
    ignore_stereo: bool,
    reorder_atoms: bool,
    use_cuikmolmaker_featurization: bool = False,
    n_workers: int = 0,
) -> list[list[PolymerDatapoint]]:
    r"""Make the :class:`PolymerDatapoint`\s for a given dataset.

    The polymer branches of the original PR's ``make_datapoints``.

    Parameters
    ----------
    polyss : list[list[str]]
        a list of ``j`` lists of ``n`` polymer SMILES strings, where ``j`` is the number of polymers
        per datapoint and ``n`` is the number of datapoints.

    See :func:`chemprop.cli.utils.parsing.make_datapoints` for the other parameters.

    Returns
    -------
    list[list[PolymerDatapoint]]
        a list of ``j`` lists of ``n`` :class:`PolymerDatapoint`\s
    """
    if use_cuikmolmaker_featurization:
        raise ValueError("cuik-molmaker featurization is not supported for polymers")

    N = len(polyss[0])

    weights = np.ones(N, dtype=np.single) if weights is None else weights
    gt_mask = [None] * N if gt_mask is None else gt_mask
    lt_mask = [None] * N if lt_mask is None else lt_mask
    n_mols = len(polyss)
    V_fss = [[None] * N] * n_mols if V_fss is None else V_fss
    E_fss = [[None] * N] * n_mols if E_fss is None else E_fss
    V_dss = [[None] * N] * n_mols if V_dss is None else V_dss

    poly_molss = [
        [
            (
                make_polymer_mol(smi.split("|")[0], keep_h, add_h, ignore_stereo, reorder_atoms),
                smi.split("|")[1:-1],
                smi.split("<")[1:],
            )
            for smi in smis
        ]
        for smis in polyss
    ]

    if X_d is None and molecule_featurizers is None:
        X_d = [None] * N
    elif molecule_featurizers is None:
        pass
    else:
        molecule_featurizers = [MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers]

        poly_descriptors = np.hstack(
            [
                np.vstack(
                    [
                        np.hstack(
                            [
                                mf(remove_wildcard_atoms(RWMol(poly[0])))
                                for mf in molecule_featurizers
                            ]
                        )
                        for poly in poly_mols
                    ]
                )
                for poly_mols in poly_molss
            ]
        )
        if X_d is None:
            X_d = poly_descriptors
        else:
            X_d = np.hstack([X_d, poly_descriptors])

    poly_data = [
        [
            PolymerDatapoint(
                mol=poly_molss[poly_idx][i][0],
                fragment_weights=poly_molss[poly_idx][i][1],
                edge_rules=poly_molss[poly_idx][i][2],
                name=polys[i],
                y=Y[i],
                weight=weights[i],
                gt_mask=gt_mask[i],
                lt_mask=lt_mask[i],
                x_d=X_d[i],
                x_phase=None,
                V_f=V_fss[poly_idx][i],
                E_f=E_fss[poly_idx][i],
                V_d=V_dss[poly_idx][i],
            )
            for i in range(N)
        ]
        for poly_idx, polys in enumerate(polyss)
    ]

    return poly_data


def build_data_from_files(
    p_data: PathLike,
    no_header_row: bool,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    splits_col: str | None,
    weight_col: str | None,
    bounded: bool,
    p_descriptors: PathLike,
    p_atom_feats: dict[int, PathLike],
    p_bond_feats: dict[int, PathLike],
    p_atom_descs: dict[int, PathLike],
    descriptor_cols: Sequence[str] | None = None,
    n_workers: int = 0,
    **featurization_kwargs: Mapping,
) -> list[list[PolymerDatapoint]]:
    """Same signature as :func:`chemprop.cli.utils.parsing.build_data_from_files`, with
    ``smiles_cols`` holding the polymer columns."""
    if rxn_cols is not None:
        raise ValueError("reaction columns are not supported with polymers")

    polyss, _, Y, weights, lt_mask, gt_mask, X_d_extra = parse_csv(
        p_data,
        smiles_cols,
        rxn_cols,
        target_cols,
        ignore_cols,
        splits_col,
        weight_col,
        descriptor_cols,
        bounded,
        no_header_row,
    )
    n_polymers = len(polyss)
    n_datapoints = len(Y)

    X_ds = load_input_feats_and_descs(p_descriptors, None, None, feat_desc="X_d")
    if X_d_extra is not None:
        if X_ds is None:
            X_ds = X_d_extra
        else:
            X_ds = np.hstack([X_ds, X_d_extra])

    V_fss = load_input_feats_and_descs(p_atom_feats, n_polymers, n_datapoints, feat_desc="V_f")
    E_fss = load_input_feats_and_descs(p_bond_feats, n_polymers, n_datapoints, feat_desc="E_f")
    V_dss = load_input_feats_and_descs(p_atom_descs, n_polymers, n_datapoints, feat_desc="V_d")

    return make_polymer_datapoints(
        polyss,
        Y,
        weights,
        lt_mask,
        gt_mask,
        X_ds,
        V_fss,
        E_fss,
        V_dss,
        n_workers=n_workers,
        **featurization_kwargs,
    )


def make_dataset(
    data: Sequence[PolymerDatapoint],
    reaction_mode: Literal[*tuple(RxnMode.keys())] = "REAC_DIFF",
    multi_hot_atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2",
    cuikmolmaker_featurization: bool = False,
    n_workers: int = 0,
):
    """Build a :class:`PolymerDataset` from :class:`PolymerDatapoint`\\s, otherwise defer to
    :func:`chemprop.cli.utils.parsing.make_dataset`."""
    # PolymerDatapoint subclasses MoleculeDatapoint, so it must be checked before chemprop's
    # make_dataset, which would build a MoleculeDataset from it
    if not isinstance(data[0], PolymerDatapoint):
        return parsing.make_dataset(
            data, reaction_mode, multi_hot_atom_featurizer_mode, cuikmolmaker_featurization, n_workers
        )

    atom_featurizer = get_multi_hot_atom_featurizer(multi_hot_atom_featurizer_mode)
    match multi_hot_atom_featurizer_mode:
        case "RIGR":
            bond_featurizer = RIGRBondFeaturizer()
        case "V1" | "V2" | "ORGANIC":
            bond_featurizer = MultiHotBondFeaturizer()
        case _:
            raise TypeError(
                f"Unsupported atom featurizer mode '{multi_hot_atom_featurizer_mode=}'!"
            )

    extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
    extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0
    featurizer = PolymerMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        extra_atom_fdim=extra_atom_fdim,
        extra_bond_fdim=extra_bond_fdim,
    )
    return PolymerDataset(data, featurizer, n_workers=n_workers)
