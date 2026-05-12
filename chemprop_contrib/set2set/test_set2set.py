import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, RegressionFFN
from chemprop.nn.transforms import UnscaleTransform
from lightning import Trainer

from chemprop_contrib.set2set import Set2Set


def test_set2set():
    
    HIDDEN_SIZE=8
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

    mp = BondMessagePassing(d_h=HIDDEN_SIZE, depth=1)
    agg = Set2Set(
        in_channels=HIDDEN_SIZE,
        processing_steps=6,
        n_layers=3
    )
    fnn = RegressionFFN(
        n_tasks=1,
        input_dim=2*HIDDEN_SIZE,
        hidden_dim=4,
        n_layers=1,
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
    test_set2set()
