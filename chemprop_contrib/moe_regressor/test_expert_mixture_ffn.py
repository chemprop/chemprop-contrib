from lightning import Trainer
import pandas as pd

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing
from chemprop.models import MPNN
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.transforms import UnscaleTransform

from chemprop_contrib import moe_regressor


def test_moe_regressor():
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    df = pd.DataFrame.from_dict(
        dict(
            smiles=["C" * i for i in range(1, 10)],
            target=list(range(1, 10)),
        )
    )
    smiles_col = "smiles"
    target = df[["target"]].values
    data = [
        MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(df[smiles_col], target)
    ]
    dataset = MoleculeDataset(data, featurizer)
    target_scaler = dataset.normalize_targets()
    output_transform = UnscaleTransform.from_standard_scaler(target_scaler)
    dataloader = build_dataloader(dataset)

    mp = BondMessagePassing(d_h=8, depth=1)
    agg = MeanAggregation()
    fnn = moe_regressor.MixtureOfExpertsRegressionFFN(
        n_experts=2,
        n_tasks=1,
        input_dim=mp.output_dim,
        hidden_dim=4,
        n_layers=1,
        gate_hidden_dim=8,
        gate_n_layers=1,
        output_transform=output_transform,
    )
    model = MPNN(
        mp,
        agg,
        fnn,
    )

    trainer = Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    test_moe_regressor()
