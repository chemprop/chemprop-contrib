import logging
import os
from pathlib import Path
from typing import List

from mcp.types import CallToolResult, TextContent

from chemprop_contrib.mcp.args import ConvertArgs
from chemprop_contrib.mcp.utils import (
    _link_artifact,
    _link_existing,
    _make_run_dir,
    _write_artifact,
    run_chemprop_command,
)

logger = logging.getLogger(__name__)


async def chemprop_convert(convert_args: ConvertArgs) -> CallToolResult:
    """
    Run `chemprop convert` to update a Chemprop model checkpoint to a newer format.

    Parameters
    ----------
    convert_args : ConvertArgs
        Conversion mode and input/output paths for model checkpoint conversion.

    Returns
    -------
    CallToolResult
        Result containing conversion status, run directory information, and converted model link.

    Notes
    -----
    The function validates the input model file exists, creates a run directory, and executes
    the chemprop convert command. On success, attempts to link the converted model file.
    On failure, returns error information with captured output for debugging.

    Supported conversions include v1_to_v2 and v2_0_to_v2_1 format updates.
    """
    logger.info("Starting Chemprop conversion...")

    if not convert_args.input_path or not os.path.exists(str(convert_args.input_path)):
        return CallToolResult(
            is_error=True,
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Input model not found: {convert_args.input_path!r}.",
                )
            ],
        )

    run_dir = _make_run_dir(
        Path(convert_args.output_path).parent.as_posix()
        if convert_args.output_path
        else None
    )

    cmd: List[str] = ["chemprop", "convert"]
    cmd += convert_args.to_cli_args()

    result = run_chemprop_command(cmd)

    stdout_res = _link_artifact(
        run_dir, "stdout.txt", result.get("stdout", ""), "text/plain"
    )
    stderr_res = _write_artifact(
        run_dir, "stderr.txt", result.get("stderr", ""), "text/plain"
    )

    # Compute default output if not provided
    converted_link = None
    if convert_args.output_path:
        converted_link = _link_existing(convert_args.output_path)
    else:
        inp = Path(convert_args.input_path)
        newver = "v2" if convert_args.conversion == "v1_to_v2" else "v2_1"
        default_out = Path.cwd() / f"{inp.stem}_{newver}.pt"
        converted_link = _link_existing(default_out)

    if result.get("success"):
        summary = (
            "✅ **Conversion completed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {result.get('return_code', 0)}\n"
            "- Converted model: attached if detected.\n"
        )
        contents = [TextContent(type="text", text=summary), stdout_res, stderr_res]
        if converted_link:
            contents.append(converted_link)
        return CallToolResult(content=contents)
    else:
        err_msg = result.get("error", "Unknown error")
        rc = result.get("return_code", "N/A")
        summary = (
            "❌ **Conversion failed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {rc}\n"
            f"- Error: {err_msg}\n"
            "- See attached stdout/stderr for details.\n"
        )
        return CallToolResult(
            is_error=True,
            content=[TextContent(type="text", text=summary), stdout_res, stderr_res],
        )
