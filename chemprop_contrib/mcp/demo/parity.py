#!/usr/bin/env python3
"""
Evaluate Chemprop predictions and create a parity plot.

The script expects the predictions file to be that output from
'chemprop predict --data-path OChemUnseen.csv'

Author: MU/TH/UR 6000 (gpt-oss:20b)
Editor: JacksonBurns - modified slightly
"""
import pathlib
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parity_plot(
    truth: np.ndarray,
    prediction: np.ndarray,
    dest: str | pathlib.Path,
    title: str = "Parity Plot",
    x_label: str = "True LogS",
    y_label: str = "Predicted LogS",
    xlim: Tuple[float, float] = (-12, 2),
    ylim: Tuple[float, float] = (-12, 2),
    gridsize: int = 70,
    cmap: str = "viridis",
) -> None:
    """Create a hexbin parity plot with an inset pie chart."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax: Axes  # type hint

    # Hexbin
    hb = ax.hexbin(
        truth,
        prediction,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Number of compounds")

    # 1:1 line
    ax.plot(xlim, xlim, "r", linewidth=1)
    # ±1 log S lines
    ax.plot(xlim, (np.array(xlim) + 1), "r--", linewidth=0.5)
    ax.plot(xlim, (np.array(xlim) - 1), "r--", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)

    # Text box with R² and MSE
    textstr = "\n".join(
        (
            f"$\\bf{{R2}}:$ {r2_score(truth, prediction):.2f}",
            f"$\\bf{{MSE}}:$ {mean_squared_error(truth, prediction):.2f}",
        )
    )
    ax.text(
        -8.55,
        -2.1,
        textstr,
        transform=ax.transData,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Inset pie chart: fraction within ±1 log S
    _frac_wn_1 = np.count_nonzero(np.abs(truth - prediction) < 1.0) / len(truth)
    sizes = [1 - _frac_wn_1, _frac_wn_1]
    ax_inset = ax.inset_axes([-12, -2, 4, 4], transform=ax.transData)
    ax_inset.pie(
        sizes,
        colors=["#ae2b27", "#4073b2"],
        startangle=360 * (_frac_wn_1 - 0.5) / 2,
        wedgeprops={"edgecolor": "black"},
        autopct="%1.f%%",
        textprops=dict(color="w"),
    )
    ax_inset.axis("equal")

    plt.savefig(dest, dpi=300)
    plt.show()


if __name__ == "__main__":
    try:
        PRED_FILE = pathlib.Path(sys.argv[1])
    except:
        print("USAGE: python parity.py </path/to/predictions.csv>")
        exit(1)

    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Could not find {PRED_FILE}")

    df = pd.read_csv(PRED_FILE)

    # --------------------------------------------------------------------------- #
    # 2. Pull the columns of interest
    # --------------------------------------------------------------------------- #
    TRUE_COL = "LogS"
    PRED_COL = "ExperimentalLogS" if "ExperimentalLogS" in df.columns else "pred_0"

    if TRUE_COL not in df.columns or PRED_COL not in df.columns:
        raise KeyError(f"Required columns not found: {TRUE_COL}, {PRED_COL}")

    y_true = df[TRUE_COL].astype(float).values
    y_pred = df[PRED_COL].astype(float).values

    # --------------------------------------------------------------------------- #
    # 3. Compute metrics
    # --------------------------------------------------------------------------- #
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "samples": len(y_true),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
    print("Regression metrics")
    for k, v in metrics.items():
        print(f"  {k:>5}: {v:.4f}")

    # --------------------------------------------------------------------------- #
    # 5. Call the plot function
    # --------------------------------------------------------------------------- #
    _out = PRED_FILE.parent.resolve() / "parity.png"
    parity_plot(
        truth=y_true,
        prediction=y_pred,
        dest=_out,
        title=PRED_FILE.parent.resolve().stem,
        x_label="True LogS",
        y_label="Predicted LogS",
    )

    print(f"\nParity plot saved to {_out}")
