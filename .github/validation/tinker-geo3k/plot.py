"""Render the recorded GEO3K run: python plot.py (requires matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "plot-data.json").read_text())
blue, orange, ink, muted = "#2563eb", "#d97706", "#172033", "#64748b"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": ink,
    "xtick.color": muted, "ytick.color": muted, "text.color": ink,
    "savefig.facecolor": "white",
})
fig, (train_ax, eval_ax) = plt.subplots(1, 2, figsize=(13.4, 7.3), gridspec_kw={"width_ratios": [1.3, 1]})
fig.subplots_adjust(left=.075, right=.97, top=.73, bottom=.29, wspace=.25)
fig.text(.075, .935, "GEO3K · multimodal RL validation", fontsize=22, weight="bold")
fig.text(.075, .885, "Qwen3-VL-30B-A3B-Instruct  ·  64 train / 32 held-out problems  ·  16 optimizer updates", color=muted)

for ax in (train_ax, eval_ax):
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#e2e8f0", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)

batches = [row["batch"] for row in data["train"]]
rewards = [row["reward"] for row in data["train"]]
rolling = [sum(rewards[max(0, i-2):i+1]) / len(rewards[max(0, i-2):i+1]) for i in range(len(rewards))]
train_ax.plot(batches, rewards, color=blue, alpha=.3, linewidth=1.3, marker="o", markersize=4, label="Batch reward · 16 answers")
train_ax.plot(batches, rolling, color=blue, linewidth=2.5, label="3-batch trailing mean")
skipped = [row for row in data["train"] if row["skipped"]]
train_ax.scatter([x["batch"] for x in skipped], [x["reward"] for x in skipped], s=66,
                 facecolors="white", edgecolors=muted, linewidths=1.4, zorder=4, clip_on=False, label="Skipped optimizer update")
train_ax.set_xlim(.5, 21.5)
train_ax.set_xticks([1, 5, 9, 13, 17, 21])
train_ax.set_xlabel("Training rollout batch", labelpad=10)
train_ax.set_ylabel("Mean binary reward", labelpad=10)
train_ax.set_title("Training reward", loc="left", fontsize=14, weight="bold", pad=32)
train_ax.text(0, 1.045, "All 21 batches, including 5 with constant group rewards", transform=train_ax.transAxes, color=muted, fontsize=9.5)
train_ax.legend(loc="upper left", bbox_to_anchor=(0, -.23), frameon=False, fontsize=9.5, ncol=1, handlelength=2.5)

eval_ax.set_title("Held-out accuracy", loc="left", fontsize=14, weight="bold", pad=32)
eval_ax.text(0, 1.045, "Only measured before and after training", transform=eval_ax.transAxes, color=muted, fontsize=9.5)
for prefix, label, color in [("eval", "Real images", blue), ("blank", "Blank-image control", orange)]:
    measurements = [data["evaluations"][f"{prefix}-{stage}"] for stage in ("before", "after")]
    values = [row["accuracy"] for row in measurements]
    eval_ax.plot([0, data["updates"]], values, color=color, marker="o", markersize=7,
                 linewidth=1.8, linestyle=(0, (3, 4)), label=label)
    for step, row in zip([0, data["updates"]], measurements):
        count = round(row["accuracy"] * row["samples"])
        eval_ax.annotate(f"{row['accuracy']:.2%}\n{count}/{row['samples']}",
                         (step, row["accuracy"]), xytext=(0, 12), textcoords="offset points",
                         ha="center", va="bottom", color=color, fontsize=11, weight="bold")
eval_ax.set_xlim(-3, 19)
eval_ax.set_xticks([0, 4, 8, 12, 16])
eval_ax.set_xlabel("Completed optimizer updates", labelpad=10)
eval_ax.legend(loc="upper left", bbox_to_anchor=(0, -.23), frameon=False, fontsize=9.5)
fig.text(.075, .055, "Settings: 1,536-token limit · rank-8 attention LoRA · learning rate 1e−4 · seed 2026", color=ink, fontsize=10)
fig.text(.075, .022, "Real-image eval truncated: 17/32 before → 14/32 after. Training batches contain different problems.", color=muted, fontsize=9.5)
fig.savefig(ROOT / "geo3k-reward-accuracy.png", dpi=180)
fig.savefig(ROOT / "geo3k-reward-accuracy.pdf")
