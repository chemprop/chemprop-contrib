import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from mcp.types import CallToolResult, TextContent

from chemprop_contrib.mcp.args import PredictArgs
from chemprop_contrib.mcp.utils import (
    _link_artifact,
    _link_existing,
    _make_run_dir,
    _write_artifact,
    run_chemprop_command,
)

logger = logging.getLogger(__name__)


def _preview_prediction_output(path: Union[str, Path], rows: int = 5) -> str:
    """
    Preview the predictions output from chemprop predict command.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the prediction output file (.csv or .pkl).
    rows : int, default=5
        Number of rows to preview from the output file.

    Returns
    -------
    str
        Preview of the prediction output as formatted text.

    """
    if not path:
        return "(no output path detected)"
    p = Path(path)
    if not p.exists():
        return "(output file not found)"

    suffix = p.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(p, nrows=rows)
            text = df.to_csv(index=False).strip()
            return text or "(file empty)"
        if suffix == ".pkl":
            obj = pd.read_pickle(p)
            if hasattr(obj, "head"):
                df = obj.head(rows)
                text = df.to_csv(index=False).strip()
                return (
                    text or f"(pickle object with shape {getattr(obj, 'shape', '?')})"
                )
            return (
                f"(pickle object of type {type(obj).__name__}; preview not printable)"
            )
        return "(unsupported extension; expected .csv or .pkl)"
    except Exception as e:
        return f"(preview unavailable: {e})"


async def chemprop_predict(predict_args: PredictArgs) -> CallToolResult:
    """
    Run `chemprop predict` using one or more pretrained models.

    Parameters
    ----------
    predict_args : PredictArgs
        All arguments for Chemprop prediction, including input data, dataloader, featurization,
        constraints, output options, and uncertainty/calibration settings.

    Returns
    -------
    CallToolResult
        Result containing prediction status, output file links, and preview data.

    Notes
    -----
    IMPORTANT: Show parameters to the user and get explicit confirmation before invoking.

    The function validates the test file exists, creates a run directory, and executes
    the chemprop predict command. On success, attempts to link the prediction output
    file and provides a preview of the results. On failure, returns error information
    with captured output for debugging.
    """
    logger.info("Starting Chemprop prediction...")

    if not predict_args.test_path or not os.path.exists(str(predict_args.test_path)):
        return CallToolResult(
            is_error=True,
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Test CSV not found: {predict_args.test_path!r}.",
                )
            ],
        )

    # No explicit output_dir in predict; use a temp run dir next to output if provided, else temp.
    preferred_root = (
        Path(predict_args.output).parent.as_posix() if predict_args.output else None
    )
    run_dir = _make_run_dir(preferred_root)

    cmd: List[str] = ["chemprop", "predict"]
    cmd += predict_args.to_cli_args()

    result = run_chemprop_command(cmd)

    stdout_res = _link_artifact(
        run_dir, "stdout.txt", result.get("stdout", ""), "text/plain"
    )
    stderr_res = _write_artifact(
        run_dir, "stderr.txt", result.get("stderr", ""), "text/plain"
    )

    # Try to link the main predictions file if we can determine it
    pred_path: Optional[Path] = None
    if predict_args.output:
        pred_path = Path(predict_args.output)
    else:
        tpath = Path(predict_args.test_path)
        pred_path = tpath.parent / f"{tpath.stem}_preds.csv"
        if not pred_path.exists():
            pred_path = None

    pred_link = _link_existing(pred_path) if pred_path else None
    preview_text = (
        _preview_prediction_output(pred_path, rows=5)
        if pred_path
        else "(no output path detected)"
    )

    if result.get("success"):
        out_line = (
            f"- Output: `{pred_path}`\n"
            if pred_path
            else "- Output: (default location; not detected)\n"
        )
        summary = (
            "✅ **Prediction completed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {result.get('return_code', 0)}\n"
            f"{out_line}"
            "- Preview (first 5 rows):\n"
            "```\n"
            f"{preview_text}\n"
            "```\n"
            "- Artifacts: stdout.txt, stderr.txt\n"
        )
        contents = [TextContent(type="text", text=summary), stdout_res, stderr_res]
        if pred_link:
            contents.append(pred_link)
        return CallToolResult(content=contents)
    else:
        err_msg = result.get("error", "Unknown error")
        rc = result.get("return_code", "N/A")
        out_line = f"- Expected output: `{pred_path}`\n" if pred_path else ""
        summary = (
            "❌ **Prediction failed.**\n\n"
            f"- Run directory: `{run_dir}`\n"
            f"- Return code: {rc}\n"
            f"- Error: {err_msg}\n"
            f"{out_line}"
            "- See attached stdout/stderr for details.\n"
        )
        contents = [TextContent(type="text", text=summary), stdout_res, stderr_res]
        if pred_link:
            contents.append(pred_link)
        return CallToolResult(is_error=True, content=contents)
