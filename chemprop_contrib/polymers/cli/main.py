"""The ``chemprop-polymer`` command: chemprop's ``train``, ``predict`` and ``fingerprint``
subcommands with a ``-p/--polymer-columns`` option, run with :func:`polymer_cli_patches`."""

from argparse import ArgumentError, Namespace
import logging
from pathlib import Path
import sys

from configargparse import ArgumentParser

import chemprop
from chemprop.cli.conf import LOG_DIR, LOG_LEVELS, NOW
from chemprop.cli.fingerprint import FingerprintSubcommand
from chemprop.cli.predict import PredictSubcommand
from chemprop.cli.train import TrainSubcommand
from chemprop.cli.utils import pop_attr
from chemprop_contrib.polymers.cli.patches import polymer_cli_patches

logger = logging.getLogger(__name__)

# (argument name, flag) combinations that the polymer workflow does not support
_UNSUPPORTED_ARGS = [
    ("reaction_columns", "--reaction-columns"),
    ("mol_target_columns", "--mol-target-columns"),
    ("atom_target_columns", "--atom-target-columns"),
    ("bond_target_columns", "--bond-target-columns"),
    ("use_cuikmolmaker_featurization", "--use-cuikmolmaker-featurization"),
    ("from_foundation", "--from-foundation"),
]


def add_polymer_args(parser: ArgumentParser) -> ArgumentParser:
    # the polymer columns are stored as the smiles columns, so that chemprop's column handling
    # (target column inference, saved splits, prediction outputs) treats them as the input columns
    data_args = next(
        (g for g in parser._action_groups if g.title == "Shared input data args"), parser
    )
    data_args.add_argument(
        "-p",
        "--polymer-columns",
        dest="smiles_columns",
        metavar="POLYMER_COLUMNS",
        nargs="+",
        help="Column names in the input CSV containing polymer SMILES strings",
    )
    return parser


def validate_polymer_args(args: Namespace):
    for key, flag in _UNSUPPORTED_ARGS:
        if getattr(args, key, None):
            raise ArgumentError(argument=None, message=f"`{flag}` is not supported for polymers.")

    if args.smiles_columns is not None and len(args.smiles_columns) > 1:
        raise ArgumentError(
            argument=None,
            message="Only one polymer column is supported, got "
            f"{len(args.smiles_columns)}: {args.smiles_columns}.",
        )


class PolymerTrainSubcommand(TrainSubcommand):
    HELP = "Train a Chemprop wMPNN model on polymers."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return add_polymer_args(super().add_args(parser))

    @classmethod
    def func(cls, args: Namespace):
        validate_polymer_args(args)
        with polymer_cli_patches():
            super().func(args)


class PolymerPredictSubcommand(PredictSubcommand):
    HELP = "Use a pretrained Chemprop wMPNN model for prediction."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return add_polymer_args(super().add_args(parser))

    @classmethod
    def func(cls, args: Namespace):
        validate_polymer_args(args)
        with polymer_cli_patches():
            super().func(args)


class PolymerFingerprintSubcommand(FingerprintSubcommand):
    HELP = "Use a pretrained Chemprop wMPNN model to calculate learned representations."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return add_polymer_args(super().add_args(parser))

    @classmethod
    def func(cls, args: Namespace):
        validate_polymer_args(args)
        with polymer_cli_patches():
            super().func(args)


SUBCOMMANDS = [PolymerTrainSubcommand, PolymerPredictSubcommand, PolymerFingerprintSubcommand]


def construct_parser():
    parser = ArgumentParser(
        description=f"chemprop-polymer runs Chemprop {chemprop.__version__} with weighted directed message passing (wD-MPNN) for polymers."
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"chemprop {chemprop.__version__}"
    )
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)

    parent = ArgumentParser(add_help=False)
    parent.add_argument(
        "--logfile",
        "--log",
        nargs="?",
        const="default",
        help=f"Path to which the log file should be written (specifying just the flag alone will automatically log to a file ``{LOG_DIR}/MODE/TIMESTAMP.log`` , where 'MODE' is the CLI mode chosen, e.g., ``{LOG_DIR}/MODE/{NOW}.log``)",
    )
    parent.add_argument("-v", action="store_true", help="Increase verbosity level to DEBUG")
    parent.add_argument(
        "-q",
        action="count",
        default=0,
        help="Decrease verbosity level to WARNING or ERROR if specified twice",
    )

    parents = [parent]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

    return parser


def main():
    parser = construct_parser()
    args = parser.parse_args()
    logfile, v_flag, q_count, mode, func = (
        pop_attr(args, attr) for attr in ["logfile", "v", "q", "mode", "func"]
    )

    if v_flag and q_count:
        parser.error("The -v and -q options cannot be used together.")

    match logfile:
        case None:
            handler = logging.StreamHandler(sys.stderr)
        case "default":
            (LOG_DIR / mode).mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(str(LOG_DIR / mode / f"{NOW}.log"))
        case _:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(logfile)

    verbosity = q_count * -1 if q_count else (1 if v_flag else 0)
    logging_level = LOG_LEVELS.get(verbosity, logging.ERROR)
    logging.basicConfig(
        handlers=[handler],
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=logging_level,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

    func(args)


if __name__ == "__main__":
    main()
