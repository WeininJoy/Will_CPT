"""Generate figures comparing allowedK and integerK L70 k-value grids.

The script searches for

    data_*/data_allowedK/L70_kvalues.npy
    data_*/data_integerK/L70_kvalues.npy

and creates the four comparisons requested in ``figures_kvalues/``.  Values in
different arrays are aligned by their zero-based array index; curves are not
interpolated when arrays have different lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILE_NAME = "L70_kvalues.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory containing data_* folders (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_kvalues"),
        help="Output directory (default: figures_kvalues)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    return parser.parse_args()


def load_data(root: Path) -> dict[str, dict[str, np.ndarray]]:
    datasets: dict[str, dict[str, np.ndarray]] = {}
    for folder in sorted(root.glob("data_*")):
        allowed_file = folder / "data_allowedK" / FILE_NAME
        integer_file = folder / "data_integerK" / FILE_NAME
        if not (allowed_file.is_file() and integer_file.is_file()):
            continue

        allowed = np.asarray(np.load(allowed_file), dtype=float).squeeze()
        integer = np.asarray(np.load(integer_file), dtype=float).squeeze()
        if allowed.ndim != 1 or integer.ndim != 1:
            raise ValueError(f"Expected one-dimensional arrays in {folder}")
        if len(allowed) != len(integer):
            raise ValueError(
                f"Within-folder lengths differ in {folder}: "
                f"allowedK={len(allowed)}, integerK={len(integer)}"
            )

        label = folder.name.removeprefix("data_")
        datasets[label] = {
            "allowed": allowed,
            "integer": integer,
            "difference": allowed - integer,
        }

    if not datasets:
        raise FileNotFoundError(
            f"No complete data_*/data_{{allowedK,integerK}}/{FILE_NAME} "
            f"pairs found below {root.resolve()}"
        )
    return datasets


def finish_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_within_folders(
    datasets: dict[str, dict[str, np.ndarray]], output: Path, dpi: int
) -> None:
    """Comparison 1: allowedK and integerK within each folder."""
    nrows = len(datasets)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(14, max(3.0 * nrows, 5.0)), squeeze=False
    )
    for row, (label, values) in enumerate(datasets.items()):
        allowed = values["allowed"]
        integer = values["integer"]
        difference = values["difference"]
        index = np.arange(len(allowed))

        ax = axes[row, 0]
        ax.plot(index, allowed, lw=1.3, label="allowedK")
        ax.plot(index, integer, "--", lw=1.1, label="integerK")
        ax.set_title(label)
        ax.set_ylabel("k value")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.7)
        ax.plot(index, difference, color="tab:red", lw=1.2)
        ax.set_title(rf"{label}: $k_{{allowed}}-k_{{integer}}$")
        ax.set_ylabel(r"$\Delta k$")

        for axis in axes[row]:
            axis.set_xlabel("array index")
            axis.grid(alpha=0.25)

    fig.suptitle("AllowedK versus integerK within each data folder", fontsize=14)
    finish_figure(fig, output, dpi)


def plot_across_folders(
    datasets: dict[str, dict[str, np.ndarray]],
    key: str,
    title: str,
    ylabel: str,
    output: Path,
    dpi: int,
) -> None:
    """Overlay one array type from every folder, aligned by array index."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, values in datasets.items():
        array = values[key]
        ax.plot(np.arange(len(array)), array, lw=1.25, label=f"{label} (n={len(array)})")
    if key == "difference":
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("array index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    finish_figure(fig, output, dpi)


def main() -> None:
    args = parse_args()
    datasets = load_data(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loaded:")
    for label, values in datasets.items():
        print(f"  {label}: {len(values['allowed'])} values")

    suffix = args.format
    plot_within_folders(
        datasets,
        args.output_dir / f"01_allowed_vs_integer_within_folders.{suffix}",
        args.dpi,
    )
    plot_across_folders(
        datasets,
        key="allowed",
        title="AllowedK comparison between data folders",
        ylabel="allowedK k value",
        output=args.output_dir / f"02_allowedK_between_folders.{suffix}",
        dpi=args.dpi,
    )
    plot_across_folders(
        datasets,
        key="integer",
        title="IntegerK comparison between data folders",
        ylabel="integerK k value",
        output=args.output_dir / f"03_integerK_between_folders.{suffix}",
        dpi=args.dpi,
    )
    plot_across_folders(
        datasets,
        key="difference",
        title=r"Difference comparison: $k_{allowed}-k_{integer}$",
        ylabel=r"$\Delta k$",
        output=args.output_dir / f"04_allowed_minus_integer_between_folders.{suffix}",
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
