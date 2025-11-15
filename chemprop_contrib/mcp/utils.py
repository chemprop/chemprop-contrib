import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.types import EmbeddedResource, TextResourceContents

logger = logging.getLogger(__name__)

_graceful_timelimit = int(int(os.environ.get("MCP_TOOL_TIMEOUT", 60)) * 0.95)  # seconds


def run_chemprop_command(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a chemprop CLI command and return results.

    Parameters
    ----------
    cmd : List[str]
        Command to execute as a list of strings (e.g., ['chemprop', 'train', '--data-path', 'data.csv'])
    cwd : Optional[str], default=None
        Working directory for command execution. If None, uses current directory.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing command execution results with keys:
        - success: bool - Whether the command executed successfully
        - stdout: str - Standard output from the command
        - stderr: str - Standard error from the command
        - return_code: int - Return code from the command
        - error: str - Error message if command failed (only present on failure)

    Notes
    -----
    Captures both stdout and stderr from the subprocess. If the command fails
    with a CalledProcessError, the error information is captured and returned.
    Unexpected exceptions are also caught and returned as error information.
    """
    try:
        logger.info(
            f"Running command: {' '.join(cmd)} with a time limit of {_graceful_timelimit}"
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
            timeout=_graceful_timelimit,
        )
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired as e:
        logger.error(
            f"Command timed out: {e}\nConsider setting MCP_TOOL_TIMEOUT to larger number when starting the MCP server."
        )
        return {
            "success": False,
            "error": f"Tool call timed out after {_graceful_timelimit} seconds. Decrease training time or increase MCP_TOOL_TIMEOUT environment variable.",
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return {
            "success": False,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "return_code": e.returncode,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"success": False, "error": str(e)}


def _make_run_dir(preferred_root: Optional[str] = None) -> Path:
    """
    Create a per-run directory under output_dir if provided, else a temp dir.

    Parameters
    ----------
    preferred_root : Optional[str], default=None
        Preferred root directory for creating the run directory. If None,
        creates a temporary directory instead.

    Returns
    -------
    Path
        Path to the created run directory with timestamp-based naming.

    Notes
    -----
    Creates a directory with timestamp-based naming format:
    - If preferred_root is provided: {preferred_root}/mcp_run_{timestamp}
    - If preferred_root is None: temp directory with prefix chemprop_mcp_{timestamp}_

    The timestamp format is YYYYMMDD_HHMMSS.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if preferred_root:
        root = Path(preferred_root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"mcp_run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix=f"chemprop_mcp_{ts}_"))
    return run_dir


def _write_artifact(
    run_dir: Path, filename: str, data: str, mime: str
) -> EmbeddedResource:
    """
    Write text data to run_dir/filename and return a proper EmbeddedResource.

    Parameters
    ----------
    run_dir : Path
        Directory where the artifact file should be written.
    filename : str
        Name of the file to create.
    data : str
        Text data to write to the file.
    mime : str
        MIME type for the file content.

    Returns
    -------
    EmbeddedResource
        EmbeddedResource with type='resource' and TextResourceContents containing
        the file URI, MIME type, and embedded text content.

    Notes
    -----
    Writes the data to a file in the specified run directory and creates an
    EmbeddedResource that includes the full text content embedded in the response.
    Uses UTF-8 encoding with error handling for invalid characters.
    """
    path = run_dir / filename
    path.write_text(data or "", encoding="utf-8", errors="replace")
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=path.resolve().as_uri(),
            mimeType=mime,
            text=path.read_text(encoding="utf-8", errors="replace"),
        ),
    )


def _link_artifact(run_dir: Path, filename: str, data: str, mime: str) -> dict:
    """
    Write text data to run_dir/filename and return a ResourceLink content block.

    Parameters
    ----------
    run_dir : Path
        Directory where the artifact file should be written.
    filename : str
        Name of the file to create.
    data : str
        Text data to write to the file.
    mime : str
        MIME type for the file content.

    Returns
    -------
    dict
        ResourceLink content block with type='resource_link', URI, MIME type,
        and filename. This avoids embedding the bytes in the result.

    Notes
    -----
    Similar to _write_artifact but returns a ResourceLink instead of an
    EmbeddedResource. This is useful when you want to reference the file
    without embedding its full content in the response.
    """
    path = run_dir / filename
    path.write_text(data or "", encoding="utf-8", errors="replace")
    return {
        "type": "resource_link",
        "uri": path.resolve().as_uri(),
        "mimeType": mime,
        "name": filename,
    }


def _link_existing(
    path: Union[str, Path], name: Optional[str] = None, mime: Optional[str] = None
) -> Optional[dict]:
    """
    Return a ResourceLink for an existing file path if it exists.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the existing file to link.
    name : Optional[str], default=None
        Custom name for the resource link. If None, uses the file's basename.
    mime : Optional[str], default=None
        MIME type for the file. If None, no MIME type is specified.

    Returns
    -------
    Optional[dict]
        ResourceLink content block if the file exists, None otherwise.

    Notes
    -----
    Creates a ResourceLink to an existing file without modifying it.
    Useful for linking to output files generated by chemprop commands.
    """
    p = Path(path)
    if p.exists():
        return {
            "type": "resource_link",
            "uri": p.resolve().as_uri(),
            "name": name or p.name,
            "mimeType": mime,
        }
    return None


def _read_existing(
    path: Union[str, Path], name: Optional[str] = None, mime: Optional[str] = None
) -> Optional[EmbeddedResource]:
    """
    Return an EmbeddedResource for an existing file path if it exists.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the existing file to embed.
    name : Optional[str], default=None
        Custom name for the resource. (Unused, for API symmetry.)
    mime : Optional[str], default=None
        MIME type for the file. If None, defaults to 'text/plain'.

    Returns
    -------
    Optional[EmbeddedResource]
        EmbeddedResource with file contents if the file exists, None otherwise.
    """
    p = Path(path)
    if p.exists():
        return EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=p.resolve().as_uri(),
                mimeType=mime or "text/plain",
                text=p.read_text(encoding="utf-8", errors="replace"),
            ),
        )
    return None
