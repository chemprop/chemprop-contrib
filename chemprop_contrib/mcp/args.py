from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class CommonArgs(BaseModel):
    """
    Arguments corresponding to Chemprop's shared input data, dataloader,
    featurization, and selected top-level Trainer options.
    """

    # --- Shared input data args ---
    smiles_columns: Optional[List[str]] = Field(
        default=None,
        description="Column names in the input CSV containing SMILES strings (uses the 0th column by default).",
    )  # -s / --smiles-columns
    reaction_columns: Optional[List[str]] = Field(
        default=None,
        description="Column names in the input CSV containing reaction SMILES in the format ``REACTANT>AGENT>PRODUCT``, where 'AGENT' is optional.",
    )  # -r / --reaction-columns
    no_header_row: bool = Field(
        default=False,
        description="Turn off using the first row in the input CSV as column names.",
    )  # --no-header-row

    # --- Dataloader args ---
    num_workers: int = Field(
        default=0,
        description="Number of workers for data loading (0 disables multiprocessing).",
    )  # -n / --num-workers
    batch_size: int = Field(
        default=64, description="Batch size for training/inference."
    )  # -b / --batch-size

    # --- Lightning Trainer top‑level ---
    accelerator: str = Field(
        default="auto",
        description="Passed through to Lightning Trainer (e.g., 'cpu', 'gpu', 'auto').",
    )  # --accelerator
    devices: str = Field(
        default="auto",
        description="Passed through to Lightning Trainer (e.g., 'auto', '1', or '0,1' for multiple GPUs).",
    )  # --devices

    # --- Featurization args ---
    rxn_mode: str = Field(
        default="REAC_DIFF",
        description="""Choices for construction of atom and bond features for reactions (case insensitive):

- ``REAC_PROD``: concatenates the reactants feature with the products feature
- ``REAC_DIFF``: concatenates the reactants feature with the difference in features between reactants and products (Default)
- ``PROD_DIFF``: concatenates the products feature with the difference in features between reactants and products
- ``REAC_PROD_BALANCE``: concatenates the reactants feature with the products feature, balances imbalanced reactions
- ``REAC_DIFF_BALANCE``: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions
- ``PROD_DIFF_BALANCE``: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions""",
    )  # --rxn-mode / --reaction-mode
    multi_hot_atom_featurizer_mode: str = Field(
        default="V2",
        description="""Choices for multi-hot atom featurization scheme. This will affect both non-reaction and reaction feturization (case insensitive):

- ``V1``: Corresponds to the original configuration employed in the Chemprop V1
- ``V2``: Tailored for a broad range of molecules, this configuration encompasses all elements in the first four rows of the periodic table, along with iodine. It is the default in Chemprop V2.
- ``ORGANIC``: This configuration is designed specifically for use with organic molecules for drug research and development and includes a subset of elements most common in organic chemistry, including H, B, C, N, O, F, Si, P, S, Cl, Br, and I.
- ``RIGR``: Modified V2 (default) featurizer using only the resonance-invariant atom and bond features.""",
    )  # --multi-hot-atom-featurizer-mode
    keep_h: bool = Field(
        default=False, description="Keep explicit hydrogens from input in the MolGraph."
    )  # --keep-h
    add_h: bool = Field(
        default=False, description="Add hydrogens to the MolGraph."
    )  # --add-h
    ignore_stereo: bool = Field(
        default=False,
        description="Ignore stereochemistry (R/S, cis/trans) in SMILES parsing.",
    )  # --ignore-stereo
    reorder_atoms: bool = Field(
        default=False,
        description="Reorder atoms based on atom‑map numbers in the RDKit molecule.",
    )  # --reorder-atoms

    molecule_featurizers: Optional[List[str]] = Field(
        default=None,
        description="Extra molecule‑level featurizers to concatenate (morgan_binary, morgan_count, rdkit_2d, v1_rdkit_2d[_normalized], charge).",
    )  # --molecule-featurizers / --features-generators
    descriptors_path: Optional[str] = Field(
        default=None,
        description="Path to NPZ with extra datapoint descriptors concatenated to learned representations.",
    )  # --descriptors-path
    descriptors_columns: Optional[List[str]] = Field(
        default=None,
        description="Column names in the input CSV containing extra datapoint descriptors, like temperature and pressure.",
    )  # --descriptors-columns

    no_descriptor_scaling: bool = Field(
        default=False, description="Disable scaling of extra datapoint descriptors."
    )  # --no-descriptor-scaling
    no_atom_feature_scaling: bool = Field(
        default=False,
        description="Disable scaling of extra atom features (pre‑message‑passing).",
    )  # --no-atom-feature-scaling
    no_atom_descriptor_scaling: bool = Field(
        default=False,
        description="Disable scaling of extra atom descriptors (post‑message‑passing).",
    )  # --no-atom-descriptor-scaling
    no_bond_feature_scaling: bool = Field(
        default=False,
        description="Disable scaling of extra bond features (pre‑message‑passing).",
    )  # --no-bond-feature-scaling
    no_bond_descriptor_scaling: bool = Field(
        default=False,
        description="Disable scaling of extra bond descriptors (post‑message‑passing).",
    )  # --no-bond-descriptor-scaling

    atom_features_path: Optional[List[str]] = Field(
        default=None,
        description="Additional atom features NPZ. May repeat or use (component_index path) pairs for multi‑component.",
    )  # --atom-features-path
    atom_descriptors_path: Optional[List[str]] = Field(
        default=None,
        description="Additional atom descriptors NPZ (post‑MPNN). Supports (component_index path) pairs.",
    )  # --atom-descriptors-path
    bond_features_path: Optional[List[str]] = Field(
        default=None,
        description="Additional bond features NPZ (pre‑MPNN). Supports (component_index path) pairs.",
    )  # --bond-features-path
    bond_descriptors_path: Optional[List[str]] = Field(
        default=None,
        description="Additional bond descriptors NPZ (post‑MPNN). Supports (component_index path) pairs.",
    )  # --bond-descriptors-path

    # --- Constraints ---
    constraints_path: Optional[str] = Field(
        default=None,
        description="CSV of constraints for atom/bond targets; rows align with input CSV order.",
    )  # --constraints-path
    constraints_to_targets: Optional[List[str]] = Field(
        default=None,
        description="Target column names (atom/bond) aligned to each constraints CSV column.",
    )  # --constraints-to-targets

    # --- Optional accelerated featurization ---
    use_cuikmolmaker_featurization: bool = Field(
        default=False,
        description="Use cuik‑molmaker for accelerated featurization and reduced memory.",
    )  # --use-cuikmolmaker-featurization

    def to_cli_args(self) -> List[str]:
        """
        Convert CommonArgs instance to command line argument list.

        Returns
        -------
        List[str]
            List of command line arguments corresponding to the CommonArgs fields.

        Notes
        -----
        Converts all non-None and non-default fields to their corresponding
        command line arguments. Boolean flags are only included if True.
        List fields are expanded as multiple arguments.
        """
        args: List[str] = []
        # Shared input data
        if self.smiles_columns:
            args += ["--smiles-columns"] + self.smiles_columns
        if self.reaction_columns:
            args += ["--reaction-columns"] + self.reaction_columns
        if self.no_header_row:
            args.append("--no-header-row")

        # Dataloader
        args += ["--num-workers", str(self.num_workers)]
        args += ["--batch-size", str(self.batch_size)]

        # Trainer device settings
        args += ["--accelerator", self.accelerator]
        args += ["--devices", self.devices]

        # Featurization
        args += ["--rxn-mode", self.rxn_mode]
        args += [
            "--multi-hot-atom-featurizer-mode",
            self.multi_hot_atom_featurizer_mode,
        ]
        if self.keep_h:
            args.append("--keep-h")
        if self.add_h:
            args.append("--add-h")
        if self.ignore_stereo:
            args.append("--ignore-stereo")
        if self.reorder_atoms:
            args.append("--reorder-atoms")
        if self.molecule_featurizers:
            args += ["--molecule-featurizers"] + self.molecule_featurizers
        if self.descriptors_path:
            args += ["--descriptors-path", str(self.descriptors_path)]
        if self.descriptors_columns:
            args += ["--descriptors-columns"] + self.descriptors_columns

        if self.no_descriptor_scaling:
            args.append("--no-descriptor-scaling")
        if self.no_atom_feature_scaling:
            args.append("--no-atom-feature-scaling")
        if self.no_atom_descriptor_scaling:
            args.append("--no-atom-descriptor-scaling")
        if self.no_bond_feature_scaling:
            args.append("--no-bond-feature-scaling")
        if self.no_bond_descriptor_scaling:
            args.append("--no-bond-descriptor-scaling")

        if self.atom_features_path:
            args += ["--atom-features-path"] + self.atom_features_path
        if self.atom_descriptors_path:
            args += ["--atom-descriptors-path"] + self.atom_descriptors_path
        if self.bond_features_path:
            args += ["--bond-features-path"] + self.bond_features_path
        if self.bond_descriptors_path:
            args += ["--bond-descriptors-path"] + self.bond_descriptors_path

        if self.constraints_path:
            args += ["--constraints-path", str(self.constraints_path)]
        if self.constraints_to_targets:
            args += ["--constraints-to-targets"] + self.constraints_to_targets

        if self.use_cuikmolmaker_featurization:
            args.append("--use-cuikmolmaker-featurization")
        return args


class TrainArgs(CommonArgs):
    """
    Arguments corresponding to Chemprop's named, transfer‑learning, MPNN,
    FFN, constrainer, training input, training, and split groups.
    """

    # --- Named / top-level ---
    config_path: Optional[str] = Field(
        default=None,
        description="Path to a YAML/JSON config; CLI flags override config values.",
    )  # --config-path
    data_path: Optional[Path] = Field(
        default=None, description="Path to input CSV with SMILES and targets."
    )  # -i / --data-path
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory for training outputs (default is generated under chemprop_training).",
    )  # -o / --output-dir / --save-dir
    remove_checkpoints: bool = Field(
        default=False,
        description="Delete intermediate checkpoint files after training completes.",
    )  # --remove-checkpoints

    # --- Transfer learning ---
    checkpoint: Optional[List[str]] = Field(
        default=None,
        description="Pretrained checkpoint(s) or directory of .ckpt/.pt files to load.",
    )  # --checkpoint
    freeze_encoder: bool = Field(
        default=False,
        description="Freeze MPNN/message‑passing layers loaded from checkpoint.",
    )  # --freeze-encoder
    model_frzn: Optional[str] = Field(
        default=None, description="(Deprecated) Load and freeze model weights."
    )  # --model-frzn
    frzn_ffn_layers: int = Field(
        default=0,
        description="Freeze the first n FFN layers (use with --freeze-encoder).",
    )  # --frzn-ffn-layers
    from_foundation: Optional[str] = Field(
        default=None,
        description="Initialize message passing from a foundation model (e.g., CHEMELEON) or local model file.",
    )  # --from-foundation

    # --- Ensembling ---
    ensemble_size: int = Field(
        default=1, description="Number of models per split to ensemble."
    )  # --ensemble-size

    # --- Message passing ---
    message_hidden_dim: int = Field(
        default=300, description="Hidden dimension for messages in the MPNN."
    )  # --message-hidden-dim
    message_bias: bool = Field(
        default=False, description="Add bias terms in message‑passing layers."
    )  # --message-bias
    depth: int = Field(
        default=3, description="Number of message‑passing steps (graph layers)."
    )  # --depth
    undirected: bool = Field(
        default=False,
        description="Use undirected message passing (sum both bond directions).",
    )  # --undirected
    dropout: float = Field(
        default=0.0, description="Dropout probability in MPNN and FFN layers."
    )  # --dropout
    mpn_shared: bool = Field(
        default=False, description="Share a single MPNN across all input molecules."
    )  # --mpn-shared
    aggregation: str = Field(
        default="norm", description="Aggregation over atoms (mean, sum, norm)."
    )  # --aggregation / --agg
    aggregation_norm: float = Field(
        default=100.0, description="Normalization factor used for 'norm' aggregation."
    )  # --aggregation-norm
    atom_messages: bool = Field(
        default=False, description="Pass messages on atoms instead of bonds."
    )  # --atom-messages
    activation: str = Field(
        default="RELU",
        description="Activation used in MPNN and FFN (e.g., RELU, GELU, SILU).",
    )  # --activation
    activation_args: Optional[List[str]] = Field(
        default=None,
        description="Optional arguments for the activation (e.g., 'arg1 arg2 key=value').",
    )  # --activation-args

    # --- FFN args ---
    ffn_hidden_dim: int = Field(
        default=300, description="Hidden dimension in the (top) FFN predictor."
    )  # --ffn-hidden-dim
    ffn_num_layers: int = Field(
        default=1, description="Number of layers in the (top) FFN predictor."
    )  # --ffn-num-layers

    # --- Extra MPNN args ---
    batch_norm: bool = Field(
        default=False, description="Enable batch normalization after aggregation."
    )  # --batch-norm
    multiclass_num_classes: int = Field(
        default=3, description="Number of classes for multiclass molecule‑level tasks."
    )  # --multiclass-num-classes

    # --- Atom FFN ---
    atom_task_weights: Optional[List[float]] = Field(
        default=None, description="Loss weights applied to all atom‑level tasks."
    )  # --atom-task-weights
    atom_ffn_hidden_dim: int = Field(
        default=300, description="Hidden dimension of the atom‑level FFN."
    )  # --atom-ffn-hidden-dim
    atom_ffn_num_layers: int = Field(
        default=1, description="Number of layers in the atom‑level FFN."
    )  # --atom-ffn-num-layers
    atom_multiclass_num_classes: int = Field(
        default=3, description="Number of classes for atom‑level multiclass tasks."
    )  # --atom-multiclass-num-classes

    # --- Bond FFN ---
    bond_task_weights: Optional[List[float]] = Field(
        default=None, description="Loss weights applied to all bond‑level tasks."
    )  # --bond-task-weights
    bond_ffn_hidden_dim: int = Field(
        default=300, description="Hidden dimension of the bond‑level FFN."
    )  # --bond-ffn-hidden-dim
    bond_ffn_num_layers: int = Field(
        default=1, description="Number of layers in the bond‑level FFN."
    )  # --bond-ffn-num-layers
    bond_multiclass_num_classes: int = Field(
        default=3, description="Number of classes for bond‑level multiclass tasks."
    )  # --bond-multiclass-num-classes

    # --- Constrainer FFN ---
    atom_constrainer_ffn_hidden_dim: int = Field(
        default=300, description="Hidden dimension of the atom constrainer FFN."
    )  # --atom-constrainer-ffn-hidden-dim
    atom_constrainer_ffn_num_layers: int = Field(
        default=1, description="Number of layers in the atom constrainer FFN."
    )  # --atom-constrainer-ffn-num-layers
    bond_constrainer_ffn_hidden_dim: int = Field(
        default=300, description="Hidden dimension of the bond constrainer FFN."
    )  # --bond-constrainer-ffn-hidden-dim
    bond_constrainer_ffn_num_layers: int = Field(
        default=1, description="Number of layers in the bond constrainer FFN."
    )  # --bond-constrainer-ffn-num-layers

    # --- Training input data ---
    weight_column: Optional[str] = Field(
        default=None, description="CSV column with per‑row weights."
    )  # -w / --weight-column
    target_columns: Optional[List[str]] = Field(
        default=None,
        description="Target column names; if omitted, uses all non‑SMILES, non‑ignored columns.",
    )  # --target-columns
    mol_target_columns: Optional[List[str]] = Field(
        default=None,
        description="Molecule‑level target columns (with molecule + atom/bond targets).",
    )  # --mol-target-columns
    atom_target_columns: Optional[List[str]] = Field(
        default=None, description="Atom‑level target columns."
    )  # --atom-target-columns
    bond_target_columns: Optional[List[str]] = Field(
        default=None, description="Bond‑level target columns."
    )  # --bond-target-columns
    ignore_columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to ignore when --target-columns is not specified.",
    )  # --ignore-columns
    no_cache: bool = Field(
        default=False,
        description="Disable caching of featurized MolGraphs at training start.",
    )  # --no-cache
    splits_column: Optional[str] = Field(
        default=None,
        description="CSV column with 'train'/'val'/'test' split assignment for each row.",
    )  # --splits-column

    # --- Training args ---
    task_type: str = Field(
        default="regression",
        description="Task type: regression, regression-mve/evidential/quantile, classification, classification-dirichlet, multiclass, multiclass-dirichlet, spectral.",
    )  # -t / --task-type
    loss_function: Optional[str] = Field(
        default=None,
        description="Loss to use (defaults to the task’s standard loss if omitted).",
    )  # -l / --loss-function
    v_kl: float = Field(
        default=0.0, description="Regularization strength for evidential loss."
    )  # --v-kl / --evidential-regularization
    eps: float = Field(
        default=1e-8, description="Epsilon used in evidential regularization."
    )  # --eps
    alpha: float = Field(
        default=0.1,
        description="Target error bound for quantile interval/pinball loss.",
    )  # --alpha
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Evaluation metrics (first metric drives early-stopping/checkpointing).",
    )  # --metrics / --metric
    tracking_metric: str = Field(
        default="val_loss",
        description="Metric to track for early stopping/checkpoints (e.g., 'val_loss', 'rmse-atom').",
    )  # --tracking-metric
    show_individual_scores: bool = Field(
        default=False, description="Print per‑target scores in addition to averages."
    )  # --show-individual-scores
    task_weights: Optional[List[float]] = Field(
        default=None, description="Loss weights applied per molecule‑level task."
    )  # --task-weights
    warmup_epochs: int = Field(
        default=2,
        description="Epochs to linearly increase LR from init_lr to max_lr before decaying to final_lr.",
    )  # --warmup-epochs
    init_lr: float = Field(
        default=1e-4, description="Initial learning rate."
    )  # --init-lr
    max_lr: float = Field(
        default=1e-3, description="Maximum learning rate."
    )  # --max-lr
    final_lr: float = Field(
        default=1e-4, description="Final learning rate."
    )  # --final-lr
    epochs: int = Field(
        default=50, description="Number of training epochs."
    )  # --epochs
    patience: Optional[int] = Field(
        default=None,
        description="Epochs to wait without improvement before early stopping.",
    )  # --patience
    grad_clip: Optional[float] = Field(
        default=None,
        description="Gradient clipping value (passed to Lightning Trainer).",
    )  # --grad-clip
    class_balance: bool = Field(
        default=False,
        description="Balance positives/negatives within each training batch (classification).",
    )  # --class-balance

    # --- Split args ---
    split: str = Field(
        default="RANDOM",
        description="Data split method: RANDOM [default], SCAFFOLD_BALANCED, RANDOM_WITH_REPEATED_SMILES, KENNARD_STONE, KMEANS.",
    )  # --split / --split-type
    split_sizes: List[float] = Field(
        default_factory=lambda: [0.8, 0.1, 0.1],
        description="Fractions for train/val/test (default 0.8/0.1/0.1).",
    )  # --split-sizes
    split_key_molecule: int = Field(
        default=0,
        description="Index of the key molecule for constrained splits; multi‑component inputs.",
    )  # --split-key-molecule
    num_replicates: int = Field(
        default=1, description="Number of replicate runs (random trials)."
    )  # --num-replicates
    save_smiles_splits: bool = Field(
        default=False,
        description="Save the SMILES comprising each train/val/test split.",
    )  # --save-smiles-splits
    splits_file: Optional[str] = Field(
        default=None,
        description="JSON with predefined splits (list of dicts with 'train'/'val'/'test' index lists or ranges).",
    )  # --splits-file
    data_seed: int = Field(
        default=0,
        description="Seed for splitting (replicate i uses seed+i). Also used for shuffling when enabled.",
    )  # --data-seed

    # --- Final top-level ---
    pytorch_seed: Optional[int] = Field(
        default=None,
        description="Seed for PyTorch randomness (e.g., weight initialization).",
    )  # --pytorch-seed

    def to_cli_args(self) -> List[str]:
        """
        Convert TrainArgs instance to command line argument list.

        Returns
        -------
        List[str]
            List of command line arguments corresponding to the TrainArgs fields.

        Raises
        ------
        ValueError
            If split_sizes does not contain exactly 3 values.

        Notes
        -----
        Converts all non-None and non-default fields to their corresponding
        command line arguments. Boolean flags are only included if True.
        List fields are expanded as multiple arguments.

        The method handles all training-related arguments including:
        - Named/top-level arguments (config, data, output paths)
        - Transfer learning settings
        - Model architecture parameters (MPNN, FFN, constrainers)
        - Training hyperparameters and optimization settings
        - Data splitting and validation configuration
        """
        args: List[str] = []
        # add common args
        args += super().to_cli_args()
        # Named / top-level
        if self.config_path:
            args += ["--config-path", str(self.config_path)]
        if self.data_path:
            args += ["--data-path", str(self.data_path)]
        if self.output_dir:
            args += ["--output-dir", str(self.output_dir)]
        if self.remove_checkpoints:
            args.append("--remove-checkpoints")

        # Transfer learning
        if self.checkpoint:
            args += ["--checkpoint"] + [str(p) for p in self.checkpoint]
        if self.freeze_encoder:
            args.append("--freeze-encoder")
        if self.model_frzn:
            args += ["--model-frzn", str(self.model_frzn)]
        if self.frzn_ffn_layers is not None:
            args += ["--frzn-ffn-layers", str(self.frzn_ffn_layers)]
        if self.from_foundation:
            args += ["--from-foundation", str(self.from_foundation)]

        # Ensemble
        args += ["--ensemble-size", str(self.ensemble_size)]

        # Message passing
        args += ["--message-hidden-dim", str(self.message_hidden_dim)]
        if self.message_bias:
            args.append("--message-bias")
        args += ["--depth", str(self.depth)]
        if self.undirected:
            args.append("--undirected")
        args += ["--dropout", str(self.dropout)]
        if self.mpn_shared:
            args.append("--mpn-shared")
        args += ["--aggregation", self.aggregation]
        args += ["--aggregation-norm", str(self.aggregation_norm)]
        if self.atom_messages:
            args.append("--atom-messages")
        args += ["--activation", self.activation]
        if self.activation_args:
            args += ["--activation-args"] + [str(x) for x in self.activation_args]

        # FFN
        args += ["--ffn-hidden-dim", str(self.ffn_hidden_dim)]
        args += ["--ffn-num-layers", str(self.ffn_num_layers)]

        # Extra MPNN
        if self.batch_norm:
            args.append("--batch-norm")
        args += ["--multiclass-num-classes", str(self.multiclass_num_classes)]

        # Atom FFN
        if self.atom_task_weights:
            args += ["--atom-task-weights"] + [str(x) for x in self.atom_task_weights]
        args += ["--atom-ffn-hidden-dim", str(self.atom_ffn_hidden_dim)]
        args += ["--atom-ffn-num-layers", str(self.atom_ffn_num_layers)]
        args += ["--atom-multiclass-num-classes", str(self.atom_multiclass_num_classes)]

        # Bond FFN
        if self.bond_task_weights:
            args += ["--bond-task-weights"] + [str(x) for x in self.bond_task_weights]
        args += ["--bond-ffn-hidden-dim", str(self.bond_ffn_hidden_dim)]
        args += ["--bond-ffn-num-layers", str(self.bond_ffn_num_layers)]
        args += ["--bond-multiclass-num-classes", str(self.bond_multiclass_num_classes)]

        # Constrainers
        args += [
            "--atom-constrainer-ffn-hidden-dim",
            str(self.atom_constrainer_ffn_hidden_dim),
        ]
        args += [
            "--atom-constrainer-ffn-num-layers",
            str(self.atom_constrainer_ffn_num_layers),
        ]
        args += [
            "--bond-constrainer-ffn-hidden-dim",
            str(self.bond_constrainer_ffn_hidden_dim),
        ]
        args += [
            "--bond-constrainer-ffn-num-layers",
            str(self.bond_constrainer_ffn_num_layers),
        ]

        # Training input
        if self.weight_column:
            args += ["--weight-column", self.weight_column]
        if self.target_columns:
            args += ["--target-columns"] + self.target_columns
        if self.mol_target_columns:
            args += ["--mol-target-columns"] + self.mol_target_columns
        if self.atom_target_columns:
            args += ["--atom-target-columns"] + self.atom_target_columns
        if self.bond_target_columns:
            args += ["--bond-target-columns"] + self.bond_target_columns
        if self.ignore_columns:
            args += ["--ignore-columns"] + self.ignore_columns
        if self.no_cache:
            args.append("--no-cache")
        if self.splits_column:
            args += ["--splits-column", self.splits_column]

        # Training
        if self.task_type:
            args += ["--task-type", self.task_type]
        if self.loss_function:
            args += ["--loss-function", self.loss_function]
        args += ["--v-kl", str(self.v_kl)]
        args += ["--eps", str(self.eps)]
        args += ["--alpha", str(self.alpha)]
        if self.metrics:
            args += ["--metrics"] + self.metrics
        if self.tracking_metric:
            args += ["--tracking-metric", self.tracking_metric]
        if self.show_individual_scores:
            args.append("--show-individual-scores")
        if self.task_weights:
            args += ["--task-weights"] + [str(x) for x in self.task_weights]
        args += ["--warmup-epochs", str(self.warmup_epochs)]
        args += ["--init-lr", str(self.init_lr)]
        args += ["--max-lr", str(self.max_lr)]
        args += ["--final-lr", str(self.final_lr)]
        args += ["--epochs", str(self.epochs)]
        if self.patience is not None:
            args += ["--patience", str(self.patience)]
        if self.grad_clip is not None:
            args += ["--grad-clip", str(self.grad_clip)]
        if self.class_balance:
            args.append("--class-balance")

        # Split
        if self.split:
            args += ["--split", self.split]
        if self.split_sizes:
            if len(self.split_sizes) != 3:
                raise ValueError("--split-sizes must be 3 floats [train val test].")
            args += ["--split-sizes"] + [str(x) for x in self.split_sizes]
        args += ["--split-key-molecule", str(self.split_key_molecule)]
        args += ["--num-replicates", str(self.num_replicates)]
        if self.save_smiles_splits:
            args.append("--save-smiles-splits")
        if self.splits_file:
            args += ["--splits-file", str(self.splits_file)]
        args += ["--data-seed", str(self.data_seed)]

        # Final
        if self.pytorch_seed is not None:
            args += ["--pytorch-seed", str(self.pytorch_seed)]

        return args


class PredictArgs(CommonArgs):
    """
    Arguments for `chemprop predict` (Named + Uncertainty/Calibration),
    in addition to those provided by CommonArgs.
    """

    # --- Named arguments ---
    test_path: str = Field(
        description="Path to input CSV file containing SMILES."
    )  # -i / --test-path
    output: Optional[str] = Field(
        default=None,
        description="Where to save predictions (.csv or .pkl). If multiple models, an additional *_individual file is written.",
    )  # -o / --output / --preds-path
    drop_extra_columns: bool = Field(
        default=False,
        description="Drop all columns except SMILES and newly added prediction columns.",
    )  # --drop-extra-columns
    model_paths: List[str] = Field(
        description="Checkpoint(s)/model file(s) or directories to use for prediction."
    )  # --model-paths / --model-path

    # --- Uncertainty + calibration ---
    cal_path: Optional[str] = Field(
        default=None, description="CSV path for calibration dataset."
    )  # --cal-path
    uncertainty_method: str = Field(
        default="none",
        description="Uncertainty estimator (none, mve, ensemble, classification, evidential-*, dropout, classification-dirichlet, multiclass-dirichlet, quantile-regression).",
    )  # --uncertainty-method
    calibration_method: Optional[str] = Field(
        default=None,
        description="Uncertainty calibration method (zscaling, zelikman-interval, mve-weighting, conformal-*, isotonic*).",
    )  # --calibration-method
    evaluation_methods: Optional[List[str]] = Field(
        default=None,
        description="Uncertainty evaluation metrics (e.g., nll, miscalibration_area, ence, spearman, conformal-coverage-*).",
    )  # --evaluation-methods
    uncertainty_dropout_p: float = Field(
        default=0.1, description="Monte Carlo dropout probability."
    )  # --uncertainty-dropout-p
    dropout_sampling_size: int = Field(
        default=10, description="Number of samples for MC dropout."
    )  # --dropout-sampling-size
    calibration_interval_percentile: float = Field(
        default=95.0, description="Percentile used by calibration methods (1,100)."
    )  # --calibration-interval-percentile
    conformal_alpha: float = Field(
        default=0.1, description="Target error rate for conformal prediction (0,1)."
    )  # --conformal-alpha

    # Calibration-time additional features/descriptors/constraints
    cal_descriptors_path: Optional[List[List[str]]] = Field(
        default=None,
        description="Descriptors NPZ for calibration (repeatable; supports [idx, path] pairs).",
    )  # --cal-descriptors-path (nargs+append)
    cal_atom_features_path: Optional[List[List[str]]] = Field(
        default=None, description="Atom features NPZ for calibration (repeatable)."
    )  # --cal-atom-features-path
    cal_atom_descriptors_path: Optional[List[List[str]]] = Field(
        default=None, description="Atom descriptors NPZ for calibration (repeatable)."
    )  # --cal-atom-descriptors-path
    cal_bond_features_path: Optional[List[List[str]]] = Field(
        default=None, description="Bond features NPZ for calibration (repeatable)."
    )  # --cal-bond-features-path
    cal_bond_descriptors_path: Optional[List[List[str]]] = Field(
        default=None, description="Bond descriptors NPZ for calibration (repeatable)."
    )  # --cal-bond-descriptors-path
    cal_constraints_path: Optional[str] = Field(
        default=None, description="Constraints CSV for calibration set (atom/bond)."
    )  # --cal-constraints-path

    def _append_repeatable(
        self, args: List[str], flag: str, values: Optional[List[List[str]]]
    ):
        """
        Append repeatable command line arguments for calibration features.

        Parameters
        ----------
        args : List[str]
            List of command line arguments to append to.
        flag : str
            The command line flag to repeat (e.g., "--cal-descriptors-path").
        values : Optional[List[List[str]]]
            List of value lists, where each inner list corresponds to one repeat of the flag.

        Notes
        -----
        Handles repeatable arguments like calibration descriptors, atom features, etc.
        Each inner list in values becomes one instance of the flag with its values.
        """
        if not values:
            return
        # Each inner list corresponds to one repeat of the flag with its values
        for seg in values:
            args += [flag] + [str(x) for x in seg]

    def to_cli_args(self) -> List[str]:
        """
        Convert PredictArgs instance to command line argument list.

        Returns
        -------
        List[str]
            List of command line arguments corresponding to the PredictArgs fields.

        Notes
        -----
        Converts all non-None and non-default fields to their corresponding
        command line arguments. Boolean flags are only included if True.
        List fields are expanded as multiple arguments.

        The method handles prediction-related arguments including:
        - Named arguments (test path, output, model paths)
        - Uncertainty estimation and calibration settings
        - Repeatable calibration feature paths
        """
        args: List[str] = ["--test-path", str(self.test_path)]
        # add common args
        args += super().to_cli_args()

        if self.output:
            args += ["--output", str(self.output)]
        if self.drop_extra_columns:
            args.append("--drop-extra-columns")

        if self.model_paths:
            args += ["--model-paths"] + [str(p) for p in self.model_paths]

        if self.cal_path:
            args += ["--cal-path", str(self.cal_path)]
        if self.uncertainty_method:
            args += ["--uncertainty-method", str(self.uncertainty_method)]
        if self.calibration_method:
            args += ["--calibration-method", str(self.calibration_method)]
        if self.evaluation_methods:
            args += ["--evaluation-methods"] + [str(x) for x in self.evaluation_methods]
        args += ["--uncertainty-dropout-p", str(self.uncertainty_dropout_p)]
        args += ["--dropout-sampling-size", str(self.dropout_sampling_size)]
        args += [
            "--calibration-interval-percentile",
            str(self.calibration_interval_percentile),
        ]
        args += ["--conformal-alpha", str(self.conformal_alpha)]

        self._append_repeatable(
            args, "--cal-descriptors-path", self.cal_descriptors_path
        )
        self._append_repeatable(
            args, "--cal-atom-features-path", self.cal_atom_features_path
        )
        self._append_repeatable(
            args, "--cal-atom-descriptors-path", self.cal_atom_descriptors_path
        )
        self._append_repeatable(
            args, "--cal-bond-features-path", self.cal_bond_features_path
        )
        self._append_repeatable(
            args, "--cal-bond-descriptors-path", self.cal_bond_descriptors_path
        )

        if self.cal_constraints_path:
            args += ["--cal-constraints-path", str(self.cal_constraints_path)]

        return args


class ConvertArgs(BaseModel):
    """
    Arguments for `chemprop convert`.
    """

    conversion: str = Field(
        default="v1_to_v2", description="Conversion: v1_to_v2 or v2_0_to_v2_1."
    )  # -c / --conversion
    input_path: str = Field(
        description="Path to a model .pt checkpoint file."
    )  # -i / --input-path
    output_path: Optional[str] = Field(
        default=None,
        description="Where to save converted model (defaults to STEM_newversion.pt in current dir).",
    )  # -o / --output-path

    def to_cli_args(self) -> List[str]:
        """
        Convert ConvertArgs instance to command line argument list.

        Returns
        -------
        List[str]
            List of command line arguments corresponding to the ConvertArgs fields.

        Notes
        -----
        Converts all non-None and non-default fields to their corresponding
        command line arguments. The input_path is always included as it's required.
        """
        args: List[str] = []
        if self.conversion:
            args += ["--conversion", str(self.conversion)]
        args += ["--input-path", str(self.input_path)]
        if self.output_path:
            args += ["--output-path", str(self.output_path)]
        return args


class FingerprintArgs(CommonArgs):
    """
    Arguments for `chemprop fingerprint` (Named),
    in addition to those provided by CommonArgs.
    """

    test_path: str = Field(
        description="Path to input CSV file containing SMILES."
    )  # -i / --test-path
    output: Optional[str] = Field(
        default=None,
        description="Output path for fingerprints (.csv or .npz). Model index appended to stem.",
    )  # -o / --output / --preds-path
    model_paths: List[str] = Field(
        description="Checkpoint(s)/model file(s) or directories for fingerprinting."
    )  # --model-paths / --model-path
    ffn_block_index: int = Field(
        default=-1,
        description="Which FFN linear layer's encoding to export (-1 means last).",
    )  # --ffn-block-index

    def to_cli_args(self) -> List[str]:
        """
        Convert FingerprintArgs instance to command line argument list.

        Returns
        -------
        List[str]
            List of command line arguments corresponding to the FingerprintArgs fields.

        Notes
        -----
        Converts all non-None and non-default fields to their corresponding
        command line arguments. The test_path and ffn_block_index are always included.
        List fields are expanded as multiple arguments.
        """
        args: List[str] = ["--test-path", str(self.test_path)]
        # add common args
        args += super().to_cli_args()
        if self.output:
            args += ["--output", str(self.output)]
        if self.model_paths:
            args += ["--model-paths"] + [str(p) for p in self.model_paths]
        args += ["--ffn-block-index", str(self.ffn_block_index)]
        return args


class HpoptArgs(TrainArgs):
    """
    Arguments for `chemprop hpopt`, extending TrainArgs with
    Chemprop HPO and Ray Tune / Hyperopt settings.
    """

    # --- Chemprop HPO ---
    search_parameter_keywords: List[str] = Field(
        default_factory=lambda: ["basic"],
        description="Keywords or individual params to search (e.g., basic, learning_rate, all, depth, dropout, ...).",
    )  # --search-parameter-keywords
    hpopt_save_dir: Optional[str] = Field(
        default=None,
        description="Directory to save hyperparameter optimization results.",
    )  # --hpopt-save-dir

    # --- Ray Tune arguments ---
    raytune_num_samples: int = Field(
        default=10, description="Number of trials to run."
    )  # --raytune-num-samples
    raytune_search_algorithm: str = Field(
        default="hyperopt", description="Search algorithm: random, hyperopt, optuna."
    )  # --raytune-search-algorithm
    raytune_trial_scheduler: str = Field(
        default="FIFO", description="Trial scheduler: FIFO, AsyncHyperBand."
    )  # --raytune-trial-scheduler
    raytune_num_workers: int = Field(
        default=1, description="Number of Ray workers."
    )  # --raytune-num-workers
    raytune_use_gpu: bool = Field(
        default=False, description="Use GPUs within Ray scaling config."
    )  # --raytune-use-gpu
    raytune_num_checkpoints_to_keep: int = Field(
        default=1, description="Number of checkpoints to keep."
    )  # --raytune-num-checkpoints-to-keep
    raytune_grace_period: int = Field(
        default=10, description="ASHA grace period."
    )  # --raytune-grace-period
    raytune_reduction_factor: int = Field(
        default=2, description="ASHA reduction factor."
    )  # --raytune-reduction-factor
    raytune_temp_dir: Optional[str] = Field(
        default=None, description="Ray temporary directory."
    )  # --raytune-temp-dir
    raytune_num_cpus: Optional[int] = Field(
        default=None, description="Total CPUs for Ray init."
    )  # --raytune-num-cpus
    raytune_num_gpus: Optional[int] = Field(
        default=None, description="Total GPUs for Ray init."
    )  # --raytune-num-gpus
    raytune_max_concurrent_trials: Optional[int] = Field(
        default=None, description="Max concurrent Ray trials."
    )  # --raytune-max-concurrent-trials

    # --- Hyperopt arguments (when search algorithm = hyperopt) ---
    hyperopt_n_initial_points: Optional[int] = Field(
        default=None, description="HyperOptSearch: number of initial points."
    )  # --hyperopt-n-initial-points
    hyperopt_random_state_seed: Optional[int] = Field(
        default=None, description="HyperOptSearch: random seed."
    )  # --hyperopt-random-state-seed

    def to_cli_args(self) -> List[str]:
        """
        Convert HpoptArgs instance to command line argument list.

        Returns
        -------
        List[str]
            List of command line arguments corresponding to the HpoptArgs fields.

        Notes
        -----
        Extends TrainArgs.to_cli_args() with hyperparameter optimization specific arguments.
        Converts all non-None and non-default fields to their corresponding
        command line arguments. Boolean flags are only included if True.
        List fields are expanded as multiple arguments.

        The method handles hyperparameter optimization arguments including:
        - Chemprop HPO settings (search parameters, save directory)
        - Ray Tune configuration (samples, algorithms, schedulers, resources)
        - Hyperopt-specific settings (initial points, random seed)
        """
        args: List[str] = super().to_cli_args()

        if self.search_parameter_keywords:
            args += ["--search-parameter-keywords"] + [
                str(x) for x in self.search_parameter_keywords
            ]
        if self.hpopt_save_dir:
            args += ["--hpopt-save-dir", str(self.hpopt_save_dir)]

        args += ["--raytune-num-samples", str(self.raytune_num_samples)]
        if self.raytune_search_algorithm:
            args += ["--raytune-search-algorithm", str(self.raytune_search_algorithm)]
        if self.raytune_trial_scheduler:
            args += ["--raytune-trial-scheduler", str(self.raytune_trial_scheduler)]
        args += ["--raytune-num-workers", str(self.raytune_num_workers)]
        if self.raytune_use_gpu:
            args.append("--raytune-use-gpu")
        args += [
            "--raytune-num-checkpoints-to-keep",
            str(self.raytune_num_checkpoints_to_keep),
        ]
        args += ["--raytune-grace-period", str(self.raytune_grace_period)]
        args += ["--raytune-reduction-factor", str(self.raytune_reduction_factor)]
        if self.raytune_temp_dir:
            args += ["--raytune-temp-dir", str(self.raytune_temp_dir)]
        if self.raytune_num_cpus is not None:
            args += ["--raytune-num-cpus", str(self.raytune_num_cpus)]
        if self.raytune_num_gpus is not None:
            args += ["--raytune-num-gpus", str(self.raytune_num_gpus)]
        if self.raytune_max_concurrent_trials is not None:
            args += [
                "--raytune-max-concurrent-trials",
                str(self.raytune_max_concurrent_trials),
            ]

        if self.hyperopt_n_initial_points is not None:
            args += ["--hyperopt-n-initial-points", str(self.hyperopt_n_initial_points)]
        if self.hyperopt_random_state_seed is not None:
            args += [
                "--hyperopt-random-state-seed",
                str(self.hyperopt_random_state_seed),
            ]

        return args
