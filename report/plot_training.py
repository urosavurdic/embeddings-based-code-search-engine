import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPORT_DIR = os.path.dirname(__file__)

csvs = {
    "train/loss_step":  ("train_loss_step.csv",  "run-fixed - train/loss_step"),
    "train/loss_epoch": ("train_loss_epoch.csv", "run-fixed - train/loss_epoch"),
    "val/loss":         ("val_loss.csv",          "run-fixed - val/loss"),
    "val/Recall@10":    ("val_recall.csv",         "run-fixed - val/Recall@10"),
    "val/MRR@10":       ("val_mrr.csv",            "run-fixed - val/MRR@10"),
    "val/NDCG@10":      ("val_ndcg.csv",           "run-fixed - val/NDCG@10"),
}

titles = {
    "train/loss_step":  "Train Loss (per step)",
    "train/loss_epoch": "Train Loss (per epoch)",
    "val/loss":         "Validation Loss",
    "val/Recall@10":    "Validation Recall@10",
    "val/MRR@10":       "Validation MRR@10",
    "val/NDCG@10":      "Validation NDCG@10",
}

colors = {
    "train/loss_step":  "#e05c5c",
    "train/loss_epoch": "#e07b5c",
    "val/loss":         "#5c7be0",
    "val/Recall@10":    "#5cc45c",
    "val/MRR@10":       "#c45cc4",
    "val/NDCG@10":      "#5cc4c4",
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Code Search Model — Training Curves (run-fixed)", fontsize=14, fontweight="bold", y=1.01)

for ax, (key, (fname, col)) in zip(axes.flat, csvs.items()):
    path = os.path.join(REPORT_DIR, fname)
    df = pd.read_csv(path)
    df = df.dropna(subset=[col])
    x = df["trainer/global_step"]
    y = df[col]

    ax.plot(x, y, color=colors[key], linewidth=1.8)
    ax.set_title(titles[key], fontsize=11, fontweight="bold")
    ax.set_xlabel("Global Step", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    if "Recall" in key or "MRR" in key or "NDCG" in key:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        final = y.iloc[-1]
        ax.annotate(f"{final:.1%}", xy=(x.iloc[-1], final),
                    xytext=(-30, 8), textcoords="offset points",
                    fontsize=9, color=colors[key], fontweight="bold")
    else:
        final = y.iloc[-1]
        ax.annotate(f"{final:.3f}", xy=(x.iloc[-1], final),
                    xytext=(-30, 8), textcoords="offset points",
                    fontsize=9, color=colors[key], fontweight="bold")

plt.tight_layout()
out = os.path.join(REPORT_DIR, "training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
