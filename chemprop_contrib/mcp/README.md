# Chemprop MCP Server

A Model Context Protocol (MCP) server for [Chemprop](https://github.com/chemprop/chemprop) molecular property prediction.
This server provides a comprehensive interface to Chemprop's CLI commands through the MCP protocol, enabling seamless integration with AI assistants and other MCP-compatible tools (see the [demo](./demo/chemprop_mcp_demo.mp4) using [gpt-oss:20b](https://openai.com/index/introducing-gpt-oss/) via [Open WebUI](https://openwebui.com/)).

https://github.com/user-attachments/assets/cbc1328e-cec1-4c38-89ef-d02831adba62

Please cite [10.26434/chemrxiv-2025-tsx5s](https://doi.org/10.26434/chemrxiv-2025-tsx5s) if you use the Chemprop MCP Server in your published work.
More information about the tool, as well as a case study, are also included in the paper.

## Installation

### `pip`

One may install this package by installing `chemprop-contrib[mcp]` via `pip` (i.e., `pip install 'chemprop-contrib[mcp]'`).

### Install from GitHub

### Prerequisites

* Python 3.11 or higher
* [Chemprop](https://github.com/chemprop/chemprop) (version 2.2.0 or higher)

### Install the Package

1. Clone this repository:

```bash
git clone https://github.com/chemprop/chemprop-contrib.git
cd chemprop_contrib/mcp
```

2. Install the package and dependencies using `uv`:

```bash
uv pip install -e .
```

3. For hyperparameter optimization support:

```bash
uv pip install -e ".[hpopt]"
```

### Verify Installation

Ensure Chemprop is properly installed and accessible:

```bash
chemprop --help
```

## Usage

### Starting the MCP Server

You can start the server in two transport modes:

#### STDIO Transport (Default)

```bash
chemprop-mcp
```

#### HTTP Transport

```bash
chemprop-mcp --transport http --host 127.0.0.1 --port 8800
```

### Adding to MCP Client Configuration

Add the server to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "chemprop-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/PATH/TO/chemprop-mcp",
        "run",
        "mcp/server.py"
      ],
      "env": {
        "MCP_TOOL_TIMEOUT": "<seconds>"
      }
    }
  }
}
```

> [!NOTE]  
> `MCP_TOOL_TIMEOUT` sets the maximum allowed execution time (in seconds) for an MCP tool. Some tools, such as training or hyperparameter optimization, may require more than the default 60 seconds.

### Environment Variables

You can configure the server using the following environment variables:

* `MCP_TRANSPORT`: Transport type (`stdio` or `http`)
* `MCP_HOST`: Host interface for HTTP transport (default: `127.0.0.1`)
* `MCP_PORT`: Port for HTTP transport (default: `8800`)
* `MCP_PATH`: URL path for HTTP transport (default: `/mcp`)

### Tips

#### Open WebUI Compatibility

To run `chemprop-mcp` within Open WebUI, follow their documentation [here](https://docs.openwebui.com/features/plugin/tools/openapi-servers/mcp).
Ultimately you will end up running a command that looks something like `mcpo --port 8000 -- python mcp/server.py` (may change based on the specifics of your setup).

In addition to setting the `MCP_TOOL_TIMEOUT` environment variable higher for long model training, one may also need to increase the timeout for open-webui separately.

#### General Usage

Models need to have a wide enough context window to see the entire JSON object used to call the functions, as well as understand the entire (long) docstring for the functions.
For example, gpt-oss:20b required a context window of `32768` for our demo.

Smaller models may need to reduce the `temperature` to encourage the model to follow the Schema and not hallucinate probable arguments.
For example, gpt-oss:20b needed the `temperature` reduced from 0.8 to 0.1 to make it actually follow the Schema.
This will likely not be an issue with larger, more expressive models.

## Available Tools

`chemprop_train`: Train a model.

`chemprop_predict`: Make predictions with a trained model.

`chemprop_convert`: Convert a trained Chemprop model from v1 to v2.

`chemprop_hpopt`: Perform hyperparameter optimization.

`chemprop_fingerprint`: Use a trained model to compute a learned representation.

## Examples for Interacting with the Tools via Prompts

To interact with the Chemprop MCP server through an AI assistant or any MCP-compatible interface, use natural language prompts. Below are some examples:

- **Using it as a Chemprop Copilot**: Could you teach me how to train a Chemprop model? What kind of parameters should I set?

- **Train on foundation model**: Fine-tune the foundation model CHEMELEON on the dataset at `<data_path>` for 30 epochs, then use the trained model to generate predictions on `<test_path>`.

- **Integrated workflow**: Perform hyperparameter optimization on the dataset at `<data_path>`. Then, using the best hyperparameter configuration, train a final model and generate predictions on `<test_path>`.
