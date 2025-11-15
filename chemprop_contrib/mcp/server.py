import argparse
import logging
import os

from fastmcp import FastMCP

from chemprop_contrib.mcp.tools.convert import chemprop_convert
from chemprop_contrib.mcp.tools.fingerprint import chemprop_fingerprint
from chemprop_contrib.mcp.tools.hpopt import chemprop_hpopt
from chemprop_contrib.mcp.tools.predict import chemprop_predict
from chemprop_contrib.mcp.tools.train import chemprop_train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp")


def create_server() -> FastMCP:
    """
    Create and configure the Chemprop MCP server with all available tools.

    Returns
    -------
    FastMCP
        Configured MCP server instance with all chemprop tools registered.

    Notes
    -----
    Registers the following tools:
    - chemprop_train: Train chemprop models
    - chemprop_predict: Make predictions with trained models
    - chemprop_convert: Convert model checkpoints between versions
    - chemprop_hpopt: Perform hyperparameter optimization
    - chemprop_fingerprint: Generate molecular fingerprints
    """
    mcp = FastMCP("Chemprop MCP Server")
    mcp.tool()(chemprop_train)
    mcp.tool()(chemprop_predict)
    mcp.tool()(chemprop_convert)
    mcp.tool()(chemprop_hpopt)
    mcp.tool()(chemprop_fingerprint)
    return mcp


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the MCP server configuration.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing transport configuration.

    Notes
    -----
    Supports the following arguments:
    - transport: Transport type ('stdio' or 'http')
    - host: Host interface for HTTP transport
    - port: Port number for HTTP transport
    - path: URL path for HTTP transport

    Environment variables are used as defaults:
    - MCP_TRANSPORT: Default transport type
    - MCP_HOST: Default host interface
    - MCP_PORT: Default port number
    - MCP_PATH: Default URL path
    """
    parser = argparse.ArgumentParser(
        description="Run Chemprop MCP Server with STDIO or Streamable HTTP transport."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport to use. Defaults to env MCP_TRANSPORT or 'stdio'.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Host interface for http transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "8800")),
        help="Port for http transport.",
    )
    parser.add_argument(
        "--path",
        default=os.getenv("MCP_PATH", "/mcp"),
        help="URL path for http transport (default '/mcp').",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the Chemprop MCP server.

    Parses command line arguments, creates the MCP server, and starts it
    with the specified transport configuration.

    Raises
    ------
    ValueError
        If an unsupported transport type is specified.

    Notes
    -----
    The server can run in two modes:
    - STDIO: Standard input/output transport for local usage
    - HTTP: HTTP transport for remote access
    """
    args = parse_args()
    mcp = create_server()

    logger.info(f"Starting Chemprop MCP Server with transport: {args.transport}")

    if args.transport == "stdio":
        logger.info("Using STDIO transport.")
        mcp.run(transport="stdio")
    elif args.transport == "http":
        logger.info(
            f"Using HTTP transport on http://{args.host}:{args.port}{args.path}"
        )
        mcp.run(transport="http", host=args.host, port=args.port, path=args.path)
    else:
        logger.error(f"Unsupported transport: {args.transport}")
        raise ValueError(f"Unsupported transport: {args.transport}")


if __name__ == "__main__":
    main()
