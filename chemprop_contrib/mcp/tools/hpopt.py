import logging
import os
from pathlib import Path
from typing import List

from mcp.types import CallToolResult, TextContent

from chemprop_contrib.mcp.args import HpoptArgs
from chemprop_contrib.mcp.utils import (
    _link_artifact,
    _make_run_dir,
    _read_existing,
    _write_artifact,
    run_chemprop_command,
)

logger = logging.getLogger(__name__)


async def chemprop_hpopt(hpopt_args: HpoptArgs) -> CallToolResult:
    """
    Run `chemprop hpopt` for hyperparameter optimization via Ray Tune.

    Parameters
    ----------
    hpopt_args : HpoptArgs
        All arguments for Chemprop hyperparameter optimization, including input data,
        dataloader, featurization, constraints, training, and Ray Tune/Hyperopt options.

    Returns
    -------
    CallToolResult
        Result containing hyperparameter optimization status, run directory information,
        and best configuration file links.

    Notes
    -----
    IMPORTANT: Show parameters and confirm with the user before invoking, as this
    may launch many trials and consume significant compute.

    The function creates a run directory and executes
    the chemprop hpopt command. On success, attempts to link the best configuration file.
    On failure, returns error information with captured output for debugging.

    Hyperparameter optimization uses Ray Tune with various search algorithms (random,
    hyperopt, optuna) to find optimal model hyperparameters.
    """
    logger.info("Starting Chemprop hyperparameter optimization...")

    if hpopt_args.hpopt_save_dir is None:
        hpopt_args.hpopt_save_dir = Path(f"chemprop_hpopt/{hpopt_args.data_path.stem}")

    run_dir = _make_run_dir(hpopt_args.hpopt_save_dir)

    cmd: List[str] = ["chemprop", "hpopt"]
    cmd += hpopt_args.to_cli_args()

    result = run_chemprop_command(cmd)

    stdout_res = _link_artifact(
        run_dir, "stdout.txt", result.get("stdout", ""), "text/plain"
    )
    stderr_res = _write_artifact(
        run_dir, "stderr.txt", result.get("stderr", ""), "text/plain"
    )

    best_cfg_link = _read_existing(Path(hpopt_args.hpopt_save_dir) / "best_config.toml")

    if result.get("success"):
        summary = (
            "✅ **Hyperparameter optimization completed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {result.get('return_code', 0)}\n"
            f"- Results dir: `{hpopt_args.hpopt_save_dir or run_dir}`\n"
            "- Attached best_config.toml if detected.\n"
        )
        contents = [TextContent(type="text", text=summary), stdout_res, stderr_res]
        if best_cfg_link:
            contents.append(best_cfg_link)
        return CallToolResult(content=contents)
    else:
        err_msg = result.get("error", "Unknown error")
        rc = result.get("return_code", "N/A")
        summary = (
            "❌ **Hyperparameter optimization failed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {rc}\n"
            f"- Error: {err_msg}\n"
            "- See attached stdout/stderr for details.\n"
        )
        return CallToolResult(
            is_error=True,
            content=[TextContent(type="text", text=summary), stdout_res, stderr_res],
        )
