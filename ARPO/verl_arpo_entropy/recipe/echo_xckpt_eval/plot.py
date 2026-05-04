"""
Comparison plot for the four (HL, LL) ping-pong combos.

Two side-by-side panels:
  - left:  val-core F1 mean (good-format only) per combo, with the n_good count
           rendered above each bar so the reader sees how many rollouts
           contributed to the F1 average.
  - right: bad-format rate per combo, on a 0..1 axis.

Combos are laid out left-to-right in the cartesian-product order
(HL1,LL1) (HL1,LL2) (HL2,LL1) (HL2,LL2) so visual symmetry across HL rows
matches the matrix interpretation discussed in the spec.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_summary(summary_dir: str, combos: list[tuple[int, int]], filename: str, title: str) -> str:
    """Read per-combo summary.json files from `summary_dir` and write a PNG.

    `combos` is a list of (hl_step, ll_step). Combo label conventions match
    main.py: f"hl{hl}_ll{ll}".
    """
    f1_means, bad_rates, labels, n_goods = [], [], [], []
    for hl, ll in combos:
        label = f"HL{hl}\nLL{ll}"
        path = Path(summary_dir) / f"hl{hl}_ll{ll}.summary.json"
        with open(path) as f:
            data = json.load(f)
        labels.append(label)
        f1_means.append(data["overall"]["f1_mean"])
        bad_rates.append(data["overall"]["bad_format_rate"])
        n_goods.append(data["overall"]["n_good"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = list(range(len(combos)))

    axes[0].bar(x, f1_means, color="#3d7eff")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("val-core F1 (good-format only)")
    axes[0].set_title("F1 mean")
    for xi, v, n in zip(x, f1_means, n_goods):
        axes[0].text(xi, v + 0.01, f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, bad_rates, color="#d93636")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("bad-format rate")
    axes[1].set_title("bad-format rate")
    for xi, v in zip(x, bad_rates):
        axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(summary_dir) / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)
