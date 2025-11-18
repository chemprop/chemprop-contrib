import logging
from typing import List

from chemprop.cli.conf import CHEMPROP_TRAIN_DIR, NOW
from mcp.types import CallToolResult, TextContent

from chemprop_contrib.mcp.args import TrainArgs
from chemprop_contrib.mcp.utils import (
    _link_artifact,
    _make_run_dir,
    _write_artifact,
    run_chemprop_command,
)

logger = logging.getLogger(__name__)


async def chemprop_train(train_args: TrainArgs) -> CallToolResult:
    """
    Train a Chemprop model - behind the scenes, uses the CLI.

    Parameters
    ----------
    train_args : TrainArgs
        All arguments for Chemprop training, including input data, dataloader, featurization,
        trainer, transfer learning, model architecture, optimization, and split options.
        The complete schema for train_args is given at the end of this docstring.

    Returns
    -------
    CallToolResult
        Result containing training status, run directory information, and artifacts.

    Notes
    -----
    IMPORTANT: You should present these arguments to the user for review and
    obtain explicit confirmation before invoking this tool (to avoid accidental long training runs).

    The function creates a timestamped run directory and executes the chemprop train command.
    On success, returns stdout and stderr as artifacts. On failure, returns error information
    along with the captured output for debugging.

    Examples
    --------

    This is an example of a properly formatted JSON that can be ued with this function. This example
    is a MINIMAL example that should be sufficient for the majority of Chemprop training - adapt it for
    your own use, as needed:

    ```json
    {
    "train_args": {
        "data_path": "/path/to/input_data.csv",
        "output_dir": "/path/to/out_dir",
        "smiles_columns": [
        "SMILES"
        ],
        "target_columns": [
        "logp"
        ],
        "batch_size": 64,
        "from_foundation": "chemeleon",  // null also acceptable often
        "message_hidden_dim": 300,
        "message_bias": false,
        "depth": 3,
        "dropout": 0,
        "mpn_shared": false,
        "aggregation": "norm",
        "aggregation_norm": 100,
        "activation": "RELU",
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 1,
        "batch_norm": false,
        "multiclass_num_classes": 3,
        "task_type": "regression",
        "loss_function": "mse",
        "metrics": [
        "mae"
        ],
        "tracking_metric": "val_loss",
        "show_individual_scores": true,
        "warmup_epochs": 2,
        "init_lr": 0.0001,
        "max_lr": 0.001,
        "final_lr": 0.0001,
        "epochs": 50,
        "patience": 5,
        "grad_clip": 0,
        "class_balance": false,
        "split": "RANDOM",
        "split_sizes": [
        0.8,
        0.1,
        0.1,
        ],
        "split_key_molecule": 0,
        "num_replicates": 1,
        "data_seed": 0,
        "pytorch_seed": 0
    }
    }
    ```

    Note that the above example is for regression - if you are doing classification or multiclass classification,
    you will need to change the related arguments for Chemprop to work properly.

    Below is an example of a properly formatted JSON showing al of the possible arguments you can specify:

    ```json
    {
    "train_args": {
        "smiles_columns": [
        "SMILES"
        ],
        "reaction_columns": [
        "Coupling-Reactions"
        ],
        "no_header_row": false,
        "num_workers": 0,
        "batch_size": 64,
        "accelerator": "auto",
        "devices": "auto",
        "rxn_mode": "REAC_DIFF",
        "multi_hot_atom_featurizer_mode": "V2",
        "keep_h": false,
        "add_h": false,
        "ignore_stereo": false,
        "reorder_atoms": false,
        "molecule_featurizers": [
        "charge"
        ],
        "descriptors_path": "/path/to/descriptors.npz",
        "descriptors_columns": [
        "temperature"
        ],
        "no_descriptor_scaling": false,
        "no_atom_feature_scaling": false,
        "no_atom_descriptor_scaling": false,
        "no_bond_feature_scaling": false,
        "no_bond_descriptor_scaling": false,
        "atom_features_path": [
        "string"
        ],
        "atom_descriptors_path": [
        "/path/to/descriptors.npz"
        ],
        "bond_features_path": [
        "/path/to/features.npz"
        ],
        "bond_descriptors_path": [
        "/path/to/descriptors.npz"
        ],
        "constraints_path": "/path/to/constraints.npz",
        "constraints_to_targets": [
        "/path/to/targets.npz"
        ],
        "use_cuikmolmaker_featurization": false,
        "config_path": "/path/to/config.toml",
        "data_path": "/path/to/input_data.csv",
        "output_dir": "/path/to/out_dir",
        "remove_checkpoints": false,
        "checkpoint": [
        "/path/to/checkpoint.ckpt"
        ],
        "freeze_encoder": false,
        "model_frzn": null,
        "frzn_ffn_layers": 0,
        "from_foundation": "chemeleon",  // null also acceptable often
        "ensemble_size": 1,
        "message_hidden_dim": 300,
        "message_bias": false,
        "depth": 3,
        "undirected": false,
        "dropout": 0,
        "mpn_shared": false,
        "aggregation": "norm",
        "aggregation_norm": 100,
        "atom_messages": false,
        "activation": "RELU",
        "activation_args": null,
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 1,
        "batch_norm": false,
        "multiclass_num_classes": 3,
        "atom_task_weights": [
        0
        ],
        "atom_ffn_hidden_dim": 300,
        "atom_ffn_num_layers": 1,
        "atom_multiclass_num_classes": 3,
        "bond_task_weights": [
        0
        ],
        "bond_ffn_hidden_dim": 300,
        "bond_ffn_num_layers": 1,
        "bond_multiclass_num_classes": 3,
        "atom_constrainer_ffn_hidden_dim": 300,
        "atom_constrainer_ffn_num_layers": 1,
        "bond_constrainer_ffn_hidden_dim": 300,
        "bond_constrainer_ffn_num_layers": 1,
        "weight_column": "string",
        "target_columns": [
        "logp"
        ],
        "mol_target_columns": [
        "logp"
        ],
        "atom_target_columns": [
        "charge"
        ],
        "bond_target_columns": [
        "strength"
        ],
        "ignore_columns": [
        "doi"
        ],
        "no_cache": false,
        "splits_column": null,
        "task_type": "regression",
        "loss_function": "mse",
        "v_kl": 0,
        "eps": 1e-8,
        "alpha": 0.1,
        "metrics": [
        "mae"
        ],
        "tracking_metric": "val_loss",
        "show_individual_scores": false,
        "task_weights": [
        1.0
        ],
        "warmup_epochs": 2,
        "init_lr": 0.0001,
        "max_lr": 0.001,
        "final_lr": 0.0001,
        "epochs": 50,
        "patience": 5,
        "grad_clip": 0,
        "class_balance": false,
        "split": "RANDOM",
        "split_sizes": [
        0.8,
        0.1,
        0.1,
        ],
        "split_key_molecule": 0,
        "num_replicates": 1,
        "save_smiles_splits": false,
        "splits_file": "/path/to/split.json",
        "data_seed": 0,
        "pytorch_seed": 0
    }
    }
    ```
    """
    logger.info("Starting Chemprop model training...")

    if train_args.output_dir is None:
        train_args.output_dir = (
            train_args.data_path.parent.resolve() / CHEMPROP_TRAIN_DIR / NOW
        )

    run_dir = _make_run_dir(train_args.output_dir)

    cmd: List[str] = ["chemprop", "train"]
    cmd += train_args.to_cli_args()

    result = run_chemprop_command(cmd)

    stdout_res = _link_artifact(
        run_dir, "stdout.txt", result.get("stdout", ""), "text/plain"
    )
    stderr_res = _write_artifact(
        run_dir, "stderr.txt", result.get("stderr", ""), "text/plain"
    )

    if result.get("success"):
        summary = (
            "✅ **Chemprop training completed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {result.get('return_code', 0)}\n"
            "- Artifacts: stdout.txt, stderr.txt\n"
        )
        logger.info(summary)
        return CallToolResult(
            content=[TextContent(type="text", text=summary), stdout_res, stderr_res]
        )
    else:
        err_msg = result.get("error", "Unknown error")
        rc = result.get("return_code", "N/A")
        summary = (
            "❌ **Chemprop training failed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {rc}\n"
            f"- Error: {err_msg}\n"
            "- See attached stdout/stderr for details.\n"
        )
        logger.info(summary)
        return CallToolResult(
            is_error=True,
            content=[TextContent(type="text", text=summary), stdout_res, stderr_res],
        )
