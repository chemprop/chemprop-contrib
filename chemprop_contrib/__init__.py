# guaranteed to work imports (have no new dependencies)
from chemprop_contrib import moe_regressor

__all__ = [
    "moe_regressor",
]

# possibly not working imports, because they have external deps that must be installed
# with their optional package

try:
    from chemprop_contrib import mcp

    __all__ += ["mcp"]
except ImportError:
    pass
