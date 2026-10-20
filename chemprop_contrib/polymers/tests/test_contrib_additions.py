"""Tests added in chemprop-contrib.

None of these tests were part of the original polymer PR to Chemprop
(https://github.com/chemprop/chemprop/pull/1214), whose tests are in cli/, integration/ and
unit/featurizers/. They cover:

- the featurizer: an empty molecule, and a copolymer with unequal monomer fractions and unequal
  bond weights in each direction
- the weighted mean that ``wMPNN`` computes when the aggregation is a ``MeanAggregation``
- the ``chemprop-polymer`` CLI wrapper: ``predict`` matches a forward pass in Python, the patches to
  chemprop's CLI are undone after a run, and options that don't apply to polymers are rejected
"""

from argparse import ArgumentError

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
import torch

from chemprop import nn
from chemprop_contrib.polymers import (
    PolymerDatapoint,
    PolymerDataset,
    PolymerMolGraphFeaturizer,
    WeightedBondMessagePassing,
    build_dataloader,
    wMPNN,
)
from chemprop_contrib.polymers.cli import patches
from chemprop_contrib.polymers.cli.main import main
from chemprop_contrib.polymers.models import _weighted_mean


@pytest.fixture
def polymer_featurizer():
    return PolymerMolGraphFeaturizer()


@pytest.fixture
def data(polymer_regression_data):
    smis, Y = polymer_regression_data

    return [PolymerDatapoint.from_smi(smi, y=y) for smi, y in zip(smis, Y)]


# featurizer


def test_no_atoms(polymer_featurizer):
    mg = polymer_featurizer((Chem.Mol(), [], [""]))

    assert mg.V.shape == (1, polymer_featurizer.atom_fdim)
    assert mg.V_w.shape == (1,)


def test_unequal_weights(polymer_featurizer):
    """Atom weights follow the monomer fractions and each bond direction gets its own weight.

    The polymer tests in unit/featurizers use a 50:50 copolymer with equal weights in both bond
    directions, so they can't tell if the fractions or the two directions are swapped. The bond
    weights here are not a physical ensemble; they are all different so that any swap is caught.
    """
    smi = "[*:1]CC[*:2].[*:3]C(Cl)C[*:4]|0.75|0.25|<1-4:0.2:0.6<2-3:0.3:0.9"
    mg = polymer_featurizer(PolymerDatapoint.from_smi(smi).polymer)

    # atoms without the wildcards: 0 C, 1 C (ethylene); 2 C, 3 Cl, 4 C (vinyl chloride)
    np.testing.assert_array_equal(mg.V_w, [0.75, 0.75, 0.25, 0.25, 0.25])
    # bonds within the monomers in both directions, then for each rule `<r1-r2:w12:w21` the bond
    # from the atom at [*:r1] to the atom at [*:r2] with weight w12 and its reverse with w21
    np.testing.assert_array_equal(
        mg.edge_index,
        [[0, 1, 2, 3, 2, 4, 0, 4, 1, 2], [1, 0, 3, 2, 4, 2, 4, 0, 2, 1]],
    )
    np.testing.assert_array_almost_equal(mg.E_w, [1, 1, 1, 1, 1, 1, 0.2, 0.6, 0.3, 0.9])


# weighted mean


def test_weighted_mean_equals_mean_for_unit_weights():
    H = torch.randn(10, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    V_w = torch.ones(10)

    torch.testing.assert_close(_weighted_mean(H, batch, V_w), nn.MeanAggregation()(H, batch))


def test_weighted_mean_uses_atom_weights(data):
    """The wMPNN fingerprint with mean aggregation divides by the sum of the atom weights"""
    mp = WeightedBondMessagePassing()
    model = wMPNN(mp, nn.MeanAggregation(), nn.RegressionFFN())
    bmg, *_ = next(iter(build_dataloader(PolymerDataset(data[:4]), shuffle=False)))

    model.eval()
    with torch.no_grad():
        H_v = mp(bmg)
        n_atoms = torch.bincount(bmg.batch).unsqueeze(1)
        sum_V_w = torch.zeros(len(bmg)).index_add_(0, bmg.batch, bmg.V_w).unsqueeze(1)
        expected = nn.MeanAggregation()(H_v, bmg.batch) * n_atoms / sum_V_w
        expected = expected * bmg.degree_of_poly.unsqueeze(1)

        torch.testing.assert_close(model.fingerprint(bmg), expected)


# CLI wrapper


@pytest.mark.CLI
@pytest.mark.usefixtures("train_dir")
def test_predict_matches_python(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        "--polymer-columns",
        "poly_chemprop_input",
        "--output",
        str(tmp_path / "preds.csv"),
        "--accelerator",
        "cpu",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    cli_preds = pd.read_csv(tmp_path / "preds.csv")["pred_0"].to_numpy()

    model = wMPNN.load_from_file(model_path, map_location="cpu")
    model.eval()
    smis = pd.read_csv(input_path)["poly_chemprop_input"]
    dset = PolymerDataset([PolymerDatapoint.from_smi(smi) for smi in smis])
    with torch.no_grad():
        preds = [model(bmg, V_d, X_d) for bmg, V_d, X_d, *_ in build_dataloader(dset, shuffle=False)]
    preds = torch.cat(preds).squeeze(1).numpy()

    np.testing.assert_allclose(cli_preds, preds, rtol=1e-5, atol=1e-5)


@pytest.mark.CLI
@pytest.mark.usefixtures("train_dir")
def test_patches_restored(monkeypatch, data_path, tmp_path):
    def current():
        return [getattr(module, name) for module, name, _ in patches._targets()]

    # importing chemprop_contrib.polymers does not patch chemprop
    originals = current()
    assert all(old is not new for old, (*_, new) in zip(originals, patches._targets()))

    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "predict",
        "-i",
        input_path,
        "--model-path",
        str(tmp_path / "missing.pt"),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(FileNotFoundError):
            main()

    assert current() == originals


@pytest.mark.CLI
@pytest.mark.usefixtures("train_dir")
@pytest.mark.parametrize(
    "extra_args",
    [
        ["--reaction-columns", "poly_chemprop_input"],
        ["--mol-target-columns", "EA vs SHE (eV)"],
        ["--use-cuikmolmaker-featurization"],
        ["--from-foundation", "CHEMELEON"],
        ["--polymer-columns", "poly_chemprop_input", "poly_chemprop_input"],
    ],
)
def test_unsupported_args(monkeypatch, data_path, extra_args):
    input_path, *_ = data_path
    args = [
        "chemprop-polymer",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--polymer-columns",
        "poly_chemprop_input",
        *extra_args,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()
