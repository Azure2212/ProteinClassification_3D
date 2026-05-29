import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSV_PATH = "/data/atran16/ProteinClassification_3D/trained_results/04012026_train_126_30/Resnet152_smth_5/trainingTracking.csv"
SAVE_DIR = os.path.dirname(CSV_PATH)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
epochs = df["epoch"].tolist()

# ── Find checkpoint-saved epochs (val_acc > all previous best) ───────────────
checkpoint_epochs = []
best = -1
for _, row in df.iterrows():
    if row["val_acc"] > best:
        best = row["val_acc"]
        checkpoint_epochs.append(int(row["epoch"]))

print(f"Checkpoint saved at epochs: {checkpoint_epochs}")

def mark_checkpoints(ax, df, y_col):
    """Overlay red dots at checkpoint epochs on the given axis."""
    ck_df = df[df["epoch"].isin(checkpoint_epochs)]
    ax.scatter(ck_df["epoch"], ck_df[y_col],
               color="red", zorder=5, s=60, label="checkpoint saved")

# ── Figure layout: 3 rows ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

topk_cols = [1, 3, 5, 10, 20]
colors     = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ── Row 0 left: Train top-k accuracy ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
for k, c in zip(topk_cols, colors):
    ax0.plot(epochs, df[f"topk{k}train_acc"] * 100, color=c, marker="o",
             markersize=3, label=f"top{k}")
mark_checkpoints(ax0, df, "topk1train_acc")
ax0.set_title("Train Top-K Accuracy")
ax0.set_xlabel("Epoch"); ax0.set_ylabel("Accuracy (%)")
ax0.legend(loc="upper left", fontsize=8)
ax0.grid(True)

# ── Row 0 right: Val top-k accuracy ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
for k, c in zip(topk_cols, colors):
    ax1.plot(epochs, df[f"topk{k}val_acc"] * 100, color=c, marker="o",
             markersize=3, label=f"top{k}")
mark_checkpoints(ax1, df, "topk1val_acc")
ax1.set_title("Val Top-K Accuracy")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True)

# ── Row 1 left: Train loss ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(epochs, df["train_loss"], color="tab:blue", marker="o", markersize=3, label="train loss")
mark_checkpoints(ax2, df, "train_loss")
ax2.set_title("Train Loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(fontsize=8)
ax2.grid(True)

# ── Row 1 right: Val loss ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(epochs, df["val_loss"], color="tab:orange", marker="o", markersize=3, label="val loss")
mark_checkpoints(ax3, df, "val_loss")
ax3.set_title("Val Loss")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
ax3.legend(fontsize=8)
ax3.grid(True)

# ── Row 2 left: Learning rate ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(epochs, df["learning_rate"], color="tab:green", marker="o", markersize=3, label="LR")
mark_checkpoints(ax4, df, "learning_rate")
ax4.set_title("Learning Rate")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("LR")
ax4.yaxis.set_major_formatter(plt.FormatStrFormatter("%.7f"))
ax4.legend(fontsize=8)
ax4.grid(True)

# ── Row 2 right: Train vs Val top-1 accuracy (combined) ──────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(epochs, df["train_acc"] * 100, color="tab:blue",   marker="o", markersize=3, label="train top1")
ax5.plot(epochs, df["val_acc"]   * 100, color="tab:orange", marker="o", markersize=3, label="val top1")
mark_checkpoints(ax5, df, "val_acc")
ax5.set_title("Train vs Val Top-1 Accuracy")
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Accuracy (%)")
ax5.legend(fontsize=8)
ax5.grid(True)

# ── Add a shared legend note for red dots ────────────────────────────────────
fig.text(0.5, 0.01,
         "Red dots = epochs where checkpoint was saved (val_acc improved)",
         ha="center", fontsize=11, color="red")

save_path = os.path.join(SAVE_DIR, "full_training_overview.png")
plt.savefig(save_path, bbox_inches="tight", dpi=150)
plt.show()
print(f"Saved to: {save_path}")
