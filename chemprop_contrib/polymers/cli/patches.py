"""Run chemprop's CLI with polymer support, without modifying chemprop.

chemprop's CLI functions look up module globals (e.g. ``build_data_from_files``, ``MPNN``) at call
time. :func:`polymer_cli_patches` swaps those globals for the polymer versions while the
``chemprop-polymer`` command runs, and restores the originals on exit. Importing
:mod:`chemprop_contrib.polymers` does not patch anything.
"""

from contextlib import contextmanager
from os import PathLike

import torch

import chemprop
from chemprop import data
from chemprop.cli import fingerprint, predict, train
from chemprop_contrib.polymers.cli import parsing
from chemprop_contrib.polymers.data import build_dataloader
from chemprop_contrib.polymers.models import wMPNN
from chemprop_contrib.polymers.nn import WeightedBondMessagePassing


def load_model(path: PathLike, multicomponent: bool = False, mol_atom_bond: bool = False) -> wMPNN:
    """Replaces :func:`chemprop.models.load_model`, which would load a :class:`wMPNN` checkpoint as
    a plain :class:`~chemprop.models.MPNN`."""
    if multicomponent or mol_atom_bond:
        raise ValueError("chemprop-polymer only supports single-polymer, molecule-level models")

    return wMPNN.load_from_file(path, map_location=torch.device("cpu"))


def _targets() -> list[tuple[object, str, object]]:
    """(module, global name, replacement) for every patched global"""
    return [
        # data loading
        (train, "build_data_from_files", parsing.build_data_from_files),
        (train, "make_dataset", parsing.make_dataset),
        (train, "build_dataloader", build_dataloader),
        (predict, "build_data_from_files", parsing.build_data_from_files),
        (predict, "make_dataset", parsing.make_dataset),
        (fingerprint, "build_data_from_files", parsing.build_data_from_files),
        (fingerprint, "make_dataset", parsing.make_dataset),
        # predict and fingerprint call `data.build_dataloader`
        (data, "build_dataloader", build_dataloader),
        # model building (`build_model`), `--checkpoint`/`--model-frzn` and pretrained scalers
        (train, "MPNN", wMPNN),
        (train, "BondMessagePassing", WeightedBondMessagePassing),
        # model loading
        (predict, "load_model", load_model),
        (fingerprint, "load_model", load_model),
        # `check_featurizer_matches_model` counts bond features only for bond message passing
        (predict, "BondMessagePassing", WeightedBondMessagePassing),
    ]


@contextmanager
def polymer_cli_patches():
    """Swap chemprop's CLI globals for the polymer versions for the duration of the context."""
    targets = _targets()
    missing = [f"{m.__name__}.{name}" for m, name, _ in targets if not hasattr(m, name)]
    if missing:
        raise RuntimeError(
            f"chemprop-polymer does not support chemprop {chemprop.__version__}: "
            f"could not find {', '.join(missing)}"
        )

    originals = [(module, name, getattr(module, name)) for module, name, _ in targets]
    try:
        for module, name, new in targets:
            setattr(module, name, new)
        yield
    finally:
        for module, name, old in originals:
            setattr(module, name, old)
