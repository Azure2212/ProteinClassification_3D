"""
Evaluate trained checkpoint on the real test set (testingDataFromProfessorSu_v2_229)
with two metrics: exact prediction + sequence-similarity (>=30% identity).

Combined from:
  - visuallization/SequentialSimilarityCheckResult.ipynb (model eval + topk hits)
  - visuallization/countCorrectPerProtein.py            (per-protein hit counts)

All output (per-image CSV, summary CSV, run log) is written to:
  visuallization/<DDMMYYYY>_test_visualization/
"""

############# Importing System Libraries #############
import sys
import os
import cv2
import csv
import datetime

############# Importing support Libraries #############
import numpy as np
import torch
import pandas as pd

project_dir = "/data/atran16/ProteinClassification_3D"

############# Importing datasets / metrics / models #############
sys.path.append(f"{project_dir}/utils/datasets")
sys.path.append(f"{project_dir}/evaluations")
sys.path.append(f"{project_dir}")

from pdb_ds import test_tf, get_classes
from SequentialSimilarity import similarity_score
from models import (
    load_Resnet,
    load_ConvNeXt,
    load_CoAtNet,
    load_EfficientNetV2,
    load_VIT_SizeT,
    load_RegNetY16GF,
    load_SwinV2B,
)

# =================================================================
# CONFIG (edit these per run)
# =================================================================
train_protein_path = f"{project_dir}/3D_PDB_5013_filter12/PNG/PNG126"
checkpoint_path    = f"{project_dir}/trained_results/04012026_train_126_30/Resnet152_smth_5/PDBRSTuan.pt"
test_root          = f"{project_dir}/testingDataFromProfessorSu_v2_229"
image_size         = (224, 224)
topk               = (1, 3, 5, 10, 20, 50)
THRESHOLD          = 30  # % identity cutoff for similarity-based hit
configs            = {"model": "Resnet152", "n_classes": None, "pretrained_path": ""}

# =================================================================
# OUTPUT DIR + LOG TEE
# =================================================================
date_str   = datetime.datetime.now().strftime("%d%m%Y")
output_dir = os.path.join(project_dir, "visuallization", f"{date_str}_test_visualization")
os.makedirs(output_dir, exist_ok=True)

log_path        = os.path.join(output_dir, "run.log")
_log_fh         = open(log_path, "w")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _log_fh)
print(f"[run] {date_str} | output_dir={output_dir}")
print(f"[run] checkpoint={checkpoint_path}")
print(f"[run] test_root={test_root}\n")

# =================================================================
# DEVICE + CLASS NAMES
# =================================================================
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names      = get_classes(train_protein_path)        # {idx: pdb_id}
class_names_inv  = {v: k for k, v in class_names.items()} # {pdb_id: idx}
configs["n_classes"] = len(class_names)
test_transform   = test_tf(image_size)
MAX_K            = max(topk)
print(f"Device: {device} | Classes: {len(class_names)}")

# =================================================================
# LOAD MODEL
# =================================================================
if "Resnet" in configs["model"]:
    model = load_Resnet(name=configs["model"], num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
elif configs["model"] == "ConvNeXt":
    model = load_ConvNeXt(num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
elif "CoAtNet" in configs["model"]:
    model = load_CoAtNet(name=configs["model"], num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
elif "EfficientNetV2" in configs["model"]:
    model = load_EfficientNetV2(name=configs["model"], num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
elif configs["model"] == "MaxViT":
    model = load_VIT_SizeT(num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
elif configs["model"] == "RegNetY16GF":
    model = load_RegNetY16GF(num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
elif configs["model"] == "SwinV2B":
    model = load_SwinV2B(num_classes=configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
else:
    raise ValueError(f"Unsupported model type: {configs['model']}")

model = model.to(device)
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state["net"])
model.eval()
print(f"Model loaded from checkpoint: {checkpoint_path}\n")

# =================================================================
# BUILD TEST ITEMS
# =================================================================
class_to_pdb = class_names  # {class_idx: pdb_id}
test_items   = []

for protein in os.listdir(test_root):
    prot_path = os.path.join(test_root, protein)
    if not os.path.isdir(prot_path):
        continue
    true_pdb   = protein.upper()
    true_label = class_names_inv.get(true_pdb, -1)
    for fname in os.listdir(prot_path):
        if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png")):
            continue
        test_items.append((os.path.join(prot_path, fname), true_label, true_pdb))

test_items.sort(key=lambda x: (x[2], int(os.path.splitext(os.path.basename(x[0]))[0])))
true_pdb_of = {img_path: true_pdb for img_path, _, true_pdb in test_items}

print(f"Total test items: {len(test_items)} from {len(set(t[2] for t in test_items))} proteins\n")

# =================================================================
# EVALUATION LOOP
# =================================================================
log_csv_path = os.path.join(output_dir, "SequentialSimilarityCheckResult.csv")
log_file     = open(log_csv_path, "w", newline="")
log_writer   = csv.writer(log_file)
log_writer.writerow(["image", "ground_truth", "best_sim", "match_via", "hit"])

y_true, y_pred = [], []
rows           = []
exact_hits     = {k: 0 for k in topk}
approx_hits    = {k: 0 for k in topk}

for img_path, true_label, true_pdb in test_items:
    img_num  = int(os.path.splitext(os.path.basename(img_path))[0])
    img_name = f"{true_pdb}/{os.path.basename(img_path)}"

    img        = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits       = model(img_tensor)
        probs        = torch.softmax(logits, dim=1).squeeze(0)
        topk_indices = torch.topk(probs, k=MAX_K).indices.tolist()

    # Professor's scan: break at first hit (>= THRESHOLD)
    best_sim, matched_pdb, first_hit_pos = 0.0, None, None
    for pos, cls in enumerate(topk_indices):
        pred_pdb = class_to_pdb[cls]
        try:
            sim = similarity_score(true_pdb, pred_pdb, verbose=0)
        except Exception as e:
            print(f"  [warn] similarity({true_pdb},{pred_pdb}) failed: {e}")
            sim = 0.0
        if sim > best_sim:
            best_sim, matched_pdb = sim, pred_pdb
        if sim >= THRESHOLD:
            first_hit_pos = pos
            break

    hit = first_hit_pos is not None
    log_writer.writerow([img_name, true_pdb, f"{best_sim:.2f}", matched_pdb, hit])
    log_file.flush()

    row = {"image": img_num, "protein": true_pdb}
    for k in topk:
        exact      = 1 if true_label in topk_indices[:k] else 0
        approx_hit = 1 if (first_hit_pos is not None and first_hit_pos < k) else 0
        exact_hits[k]  += exact
        approx_hits[k] += approx_hit
        row[f"exact_predictTop{k}"]   = exact
        row[f"countSimilarityTop{k}"] = approx_hit
    rows.append(row)

    pred_class = true_label if row[f"countSimilarityTop{topk[-1]}"] else probs.argmax().item()
    y_true.append(true_label)
    y_pred.append(pred_class)

    print(f"[{img_num:>3}] {img_name:<16}  best_sim@top{MAX_K}={best_sim:5.1f}%  match_via={matched_pdb}  hit={hit}")

log_file.close()
print(f"\nSaved per-image log -> {log_csv_path}")

# =================================================================
# TOPK SUMMARY (exact vs approx)
# =================================================================
n = len(y_true)
print("\n" + "=" * 70)
for k in topk:
    print(f"Exact  Top-{k:<4}: {exact_hits[k]}/{n} ({exact_hits[k]/n:.2%})  |  "
          f"Approx Top-{k:<4} (>={THRESHOLD}% id): {approx_hits[k]}/{n} ({approx_hits[k]/n:.2%})")

# =================================================================
# SAVE FULL DATAFRAME (per-image exact + similarity per k)
# =================================================================
cols_order = ["image", "protein"]
for k in topk:
    cols_order += [f"exact_predictTop{k}", f"countSimilarityTop{k}"]

df = pd.DataFrame(rows, columns=cols_order)
df["image"]     = df["protein"].astype(str) + "/" + df["image"].astype(str) + ".png"
df.index        = range(1, len(df) + 1)
df.index.name   = "order"
for k in topk:
    df[f"exact_predictTop{k}"]   = df[f"exact_predictTop{k}"].astype(bool)
    df[f"countSimilarityTop{k}"] = df[f"countSimilarityTop{k}"].astype(bool)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
print("\n" + str(df))

csv_path = os.path.join(output_dir, "ExactNSimilarityCheckResult.csv")
df.to_csv(csv_path)
print(f"\nSaved to {csv_path}")


# =================================================================
# PER-PROTEIN ACCURACY (from countCorrectPerProtein.py)
# =================================================================
def count_per_protein(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    df["hit"] = df["hit"].astype(str).str.strip().str.lower() == "true"

    grouped = (
        df.groupby("ground_truth")
          .agg(correct=("hit", "sum"), total=("hit", "size"))
          .sort_index()
    )
    total_correct = int(grouped["correct"].sum())
    total_count   = int(grouped["total"].sum())

    print(f"\nPer-protein accuracy ({len(grouped)} proteins):\n")
    for protein, row in grouped.iterrows():
        c, t = int(row["correct"]), int(row["total"])
        print(f"  Protein {protein}: {c}/{t}  ({c/t:.1%})")

    print(f"\nOverall: {total_correct}/{total_count}  ({total_correct/total_count:.1%})")


count_per_protein(log_csv_path)

# =================================================================
# CLEANUP
# =================================================================
print(f"\n[run] done. log -> {log_path}")
_log_fh.close()
