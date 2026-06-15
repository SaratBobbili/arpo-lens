"""Convergence comparison across algorithms from local W&B `.wandb` files.

Reads the LevelDB log records inside each run directory's `run-*.wandb` (no W&B
cloud needed), pulls a single scalar history, merges resumed runs per experiment
(last writer wins per step), truncates to the shortest run, and renders a
paper-ready PNG/PDF/CSV.

Edit the CONFIG block below and run:
    python -m recipe.echo.plot_convergence
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

wandb._assert_is_internal_process = True
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

# =============================================================================
# CONFIG — edit this block for each model-size run
# =============================================================================

MODEL_SIZE = "3B"

# Per-algorithm config: wandb checkpoint dir + step-0 metric value (hardcoded).
ALGORITHMS = {
    "ECHO": {
        "label": "ECHO (ours)",
        "color": "#1f77b4",
        "wandb_dir": Path(
            "/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/"
            "checkpoints/echo3B-rerun-entropy-coeff-0-penalty-0.1/wandb"
        ),
        "step0_value": None,  # TODO: set float, e.g. 0.352
    },
    "ARPO": {
        "label": "ARPO",
        "color": "#888888",
        "wandb_dir": Path(
            "/scratch/user/saratb_tamu.edu/research/arpo-lens/"
            "ARPO/checkpoints/arpo/wandb"
        ),
        "step0_value": None,  # TODO: set float, e.g. 0.341
    },
}

# Which algorithms to include in the plot (must be keys of ALGORITHMS above).
ACTIVE_ALGORITHMS = ["ECHO", "ARPO"]

METRIC = "val-aux/DR_grpo_mix/f1_score/mean@1"
STEP_KEY = "training/global_step"
OUTPUT_DIR = Path(__file__).resolve().parent / "convergence_plot"
OUTPUT_STEM = f"convergence_{MODEL_SIZE}_f1"

# =============================================================================
# Wandb DataStore loader (monkey-patched to open the .wandb file read-only)
# =============================================================================


def _open_ro(self: datastore.DataStore, fname: str) -> None:
    self._fname = fname
    self._fp = open(fname, "rb")
    self._index = 0
    self._size_bytes = os.stat(fname).st_size
    self._opened_for_scan = True
    self._read_header()


datastore.DataStore.open_for_scan = _open_ro


def _flat_key(item) -> str:
    return "/".join(item.nested_key) if item.nested_key else item.key


def read_run_history(wandb_file: Path, metric: str, step_key: str) -> dict[int, float]:
    """Scan one `run-*.wandb` file and return {global_step: metric_value}."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    history: dict[int, float] = {}
    pbar = tqdm(desc=f"scan {wandb_file.parent.name}", unit="rec", leave=False)
    while True:
        data = ds.scan_data()
        if data is None:
            break
        pbar.update(1)
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        items = {_flat_key(it): it.value_json for it in rec.history.item}
        if metric not in items or step_key not in items:
            continue
        history[int(json.loads(items[step_key]))] = float(json.loads(items[metric]))
    pbar.close()
    return history


def collect_experiment(wandb_dir: Path, metric: str, step_key: str) -> dict[int, float]:
    """Merge all `run-*/run-*.wandb` files under `wandb_dir`, latest-mtime wins."""
    run_files = sorted(wandb_dir.glob("run-*/run-*.wandb"), key=lambda p: p.stat().st_mtime)
    assert run_files, f"no run-*.wandb under {wandb_dir}"
    merged: dict[int, float] = {}
    for f in run_files:
        merged.update(read_run_history(f, metric, step_key))
    return merged


# =============================================================================
# Plot
# =============================================================================


def main() -> None:
    series = []
    for key in ACTIVE_ALGORITHMS:
        cfg = ALGORITHMS[key]
        h = collect_experiment(cfg["wandb_dir"], METRIC, STEP_KEY)
        if cfg["step0_value"] is not None:
            h[0] = cfg["step0_value"]
        steps = np.array(sorted(h))
        values = np.array([h[s] for s in steps])
        series.append({**cfg, "steps": steps, "values": values})
        print(f"{cfg['label']}: {len(steps)} pts, steps {steps.min()}..{steps.max()}")

    cutoff = min(s["steps"].max() for s in series)
    print(f"truncating both to step <= {cutoff}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for s in series:
        m = s["steps"] <= cutoff
        x, y = s["steps"][m], s["values"][m]
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.8, color=s["color"], label=s["label"])

    echo_max = series[0]["values"][series[0]["steps"] <= cutoff].max()
    arpo_x = series[1]["steps"][series[1]["steps"] <= cutoff]
    arpo_y = series[1]["values"][series[1]["steps"] <= cutoff]
    crossings = np.where(arpo_y >= echo_max * 0.95)[0]
    if crossings.size:
        ax.axhline(echo_max, color=series[0]["color"], linestyle="--", linewidth=1.0, alpha=0.5)
        first_step_arpo = int(arpo_x[crossings[0]])
        first_step_echo = int(series[0]["steps"][np.argmax(series[0]["values"] >= echo_max * 0.95)])
        print(f"ECHO peaks at f1={echo_max:.3f}; ARPO first reaches 0.95 of peak at step {first_step_arpo} vs ECHO step {first_step_echo}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("F1-score (mean@1)")
    ax.set_title("Training convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(left=0, right=cutoff + 1)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    csv = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    with open(csv, "w") as f:
        cols = ["step"] + [s["label"] for s in series]
        f.write(",".join(cols) + "\n")
        all_steps = sorted({int(x) for s in series for x in s["steps"] if x <= cutoff})
        for st in all_steps:
            row = [str(st)]
            for s in series:
                idx = np.where(s["steps"] == st)[0]
                row.append(f"{s['values'][idx[0]]:.6f}" if idx.size else "")
            f.write(",".join(row) + "\n")

    print(f"wrote {png}\nwrote {pdf}\nwrote {csv}")


if __name__ == "__main__":
    main()
