from chemprop_contrib import __all__

def main():
    if "mcp" not in __all__:
        raise RuntimeError("mcp is not available - install it with `pip install 'chemprop_contrib[mcp]'`")
    else:
        from chemprop_contrib.mcp.server import main

        main()
