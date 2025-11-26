import pytest
from fastmcp import Client

from chemprop_contrib.mcp.args import TrainArgs
from chemprop_contrib.mcp.server import create_server


@pytest.mark.asyncio
async def test_chemprop_train_basic_call(tmp_path):
    """Basic call to chemprop_train tool with minimal TrainArgs."""

    # Create the MCP server and fetch the tool
    mcp = create_server()
    tools = await mcp.get_tools()
    assert "chemprop_train" in tools

    # Minimal valid train args
    train_args = dict(
        train_args=TrainArgs(
            data_path=str(tmp_path / "dummy.csv"),
            output_dir=str(tmp_path / "out"),
            smiles_columns=["SMILES"],
            target_columns=["y"],
            batch_size=4,
            task_type="regression",
            metrics=["mae"],
            epochs=3,
            split="RANDOM",
            split_sizes=[0.8, 0.1, 0.1],
            # required for GHA - mac runner has an MPS bug
            accelerator="cpu",
            devices="1",
        )
    )

    # Fake CSV input so TrainArgs doesn't choke on nonexistent file
    (tmp_path / "dummy.csv").write_text("SMILES,y\n" + ("C,1\n" * 20))

    # Connect directly to the mcp_server instance using the in-memory transport
    async with Client(mcp) as client:
        result = await client.call_tool("chemprop_train", train_args)

    assert not result.is_error
    assert len(result.content) >= 1

    # Check summary text appears
    summary = result.content[0].text
    assert "Chemprop training completed" in summary
