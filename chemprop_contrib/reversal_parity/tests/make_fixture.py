"""Generate ``test_data.csv`` for the reversal_parity tests.

Usage::

    python make_fixture.py [path/to/rxn.csv]

The source defaults to ``../chemprop/tests/data/regression/rxn/rxn.csv``, i.e. a chemprop
checkout next to the directory the script is run from. ``test_data.csv`` is written next to
this script with columns ``smiles,ea,y_odd,y_even``.

``y_odd`` and ``y_even`` are derived from the Crippen logP of the reactant and product sides.
They are exactly odd and even under reaction reversal, and chemically meaningless.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

from chemprop.utils import make_mol
from rdkit.Chem import Crippen

DEFAULT_SOURCE = Path("../chemprop/tests/data/regression/rxn/rxn.csv")
OUTPUT = Path(__file__).with_name("test_data.csv")
DECIMALS = 4


def derived_targets(smi: str) -> tuple[float, float]:
    """Return ``(y_odd, y_even)`` for a reaction SMILES; each side is one Mol."""
    rct, pdt = (
        Crippen.MolLogP(make_mol(side, keep_h=True, add_h=False)) for side in smi.split(">>")
    )
    return round(pdt - rct, DECIMALS), round(pdt + rct, DECIMALS)


def main(source: Path) -> None:
    with open(source, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["smiles", "ea", "y_odd", "y_even"])
        for row in rows:
            writer.writerow([row["smiles"], row["ea"], *derived_targets(row["smiles"])])
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE)
