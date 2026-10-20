from pathlib import Path
import shutil

import pandas as pd
import pytest

_DATA_DIR = Path(__file__).parent / "data"
_POLY = pd.read_csv(_DATA_DIR / "test_polymer.csv")


@pytest.fixture
def data_dir():
    return _DATA_DIR


@pytest.fixture
def polymer_regression_data():
    smis = _POLY["smiles"].to_list()
    Y = _POLY["EA vs SHE (eV)"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def train_dir(monkeypatch, tmp_path):
    # keep CLI runs without --save-dir out of the working directory
    monkeypatch.setattr("chemprop.cli.train.CHEMPROP_TRAIN_DIR", tmp_path / "chemprop_training")


@pytest.fixture
def data_path(data_dir, tmp_path):
    # predict and fingerprint write their outputs next to the input file by default
    input_path = tmp_path / "dataset-poly_chemprop_small.csv"
    shutil.copy(data_dir / "regression" / "polymer" / "dataset-poly_chemprop_small.csv", input_path)
    return (str(input_path),)


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "regression" / "polymer" / "example_model_v2_regression_polymer.ckpt")


@pytest.fixture
def config_path(data_dir):
    return str(data_dir / "regression" / "polymer" / "config.toml")
