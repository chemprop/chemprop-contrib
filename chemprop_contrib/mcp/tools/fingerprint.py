import logging
import os
from pathlib import Path
from typing import List

from mcp.types import CallToolResult, TextContent

from chemprop_contrib.mcp.args import FingerprintArgs
from chemprop_contrib.mcp.utils import (
    _link_artifact,
    _link_existing,
    _make_run_dir,
    _write_artifact,
    run_chemprop_command,
)

logger = logging.getLogger(__name__)


async def chemprop_fingerprint(fp_args: FingerprintArgs) -> CallToolResult:
    """
    Run `chemprop fingerprint` to export learned representations (encodings).

    Parameters
    ----------
    fp_args : FingerprintArgs
        All arguments for Chemprop fingerprinting, including input data, dataloader,
        featurization, constraints, output options, and fingerprinting-specific settings.

    Returns
    -------
    CallToolResult
        Result containing fingerprinting status, run directory information, and output file links.

    Notes
    -----
    The function validates the test file exists, creates a run directory, and executes
    the chemprop fingerprint command. On success, attempts to link the fingerprint output
    files. On failure, returns error information with captured output for debugging.

    Fingerprinting extracts learned molecular representations from trained chemprop models,
    which can be used for downstream tasks like similarity analysis or transfer learning.
    """
    logger.info("Starting Chemprop fingerprint...")

    if not fp_args.test_path or not os.path.exists(str(fp_args.test_path)):
        return CallToolResult(
            is_error=True,
            content=[
                TextContent(
                    type="text", text=f"❌ Test CSV not found: {fp_args.test_path!r}."
                )
            ],
        )

    run_dir = _make_run_dir(
        Path(fp_args.output).parent.as_posix() if fp_args.output else None
    )

    cmd: List[str] = ["chemprop", "fingerprint"]
    cmd += fp_args.to_cli_args()

    result = run_chemprop_command(cmd)

    stdout_res = _link_artifact(
        run_dir, "stdout.txt", result.get("stdout", ""), "text/plain"
    )
    stderr_res = _write_artifact(
        run_dir, "stderr.txt", result.get("stderr", ""), "text/plain"
    )

    # Attempt to link a main output if explicitly provided
    fp_link = _link_existing(fp_args.output) if fp_args.output else None

    if result.get("success"):
        summary = (
            "✅ **Fingerprinting completed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {result.get('return_code', 0)}\n"
            "- Outputs: one file per model (index appended to stem); attached if detected.\n"
        )
        contents = [TextContent(type="text", text=summary), stdout_res, stderr_res]
        if fp_link:
            contents.append(fp_link)
        return CallToolResult(content=contents)
    else:
        err_msg = result.get("error", "Unknown error")
        rc = result.get("return_code", "N/A")
        summary = (
            "❌ **Fingerprinting failed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {rc}\n"
            f"- Error: {err_msg}\n"
            "- See attached stdout/stderr for details.\n"
        )
        return CallToolResult(
            is_error=True,
            content=[TextContent(type="text", text=summary), stdout_res, stderr_res],
        )
