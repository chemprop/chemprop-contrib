from pathlib import Path

import pandas as pd
import pytest
from chemprop_contrib.sklearn_integration import (
    ChempropMulticomponentTransformer,
    ChempropRegressor,
)
from sklearn.pipeline import Pipeline


@pytest.fixture
def rxn_mol_regression_data():
    df = pd.read_csv(Path(__file__).parent / "rxn+mol.csv")
    rxns = df["rxn_smiles"].to_list()
    smis = df["solvent_smiles"].to_list()
    Y = df["target"].to_numpy().reshape(-1, 1)

    return rxns, smis, Y


def test_sklearn_pipeline(rxn_mol_regression_data, tmp_path):
    sklearnPipeline = Pipeline(
        [
            (
                "featurizer",
                ChempropMulticomponentTransformer(
                    component_types=["molecule", "reaction"]
                ),
            ),
            ("regressor", ChempropRegressor(epochs=100)),
        ]
    )
    rxns, smis, Y = rxn_mol_regression_data
    sklearnPipeline.fit(X=[smis, rxns], y=Y)
    score = sklearnPipeline.score(X=[smis, rxns], y=Y)
    assert score[0] < 1
    sklearnPipeline["regressor"].save_model(tmp_path)
