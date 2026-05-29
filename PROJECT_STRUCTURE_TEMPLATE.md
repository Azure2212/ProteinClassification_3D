# Project Structure Template — Deep Learning Classification

A reusable template inspired by `ProteinClassification_AnhTuanTran`. Use this as a starting blueprint when bootstrapping a new image classification / deep learning project.

---

## 1. Top-Level Layout

```
<ProjectName>/
├── 3D_PDB_Dataset/              # Raw + processed dataset (keep out of git if large)
├── evaluations/                 # Metrics, charts, confusion matrices
├── models/                      # Model architectures / loaders
├── trainer/                     # Training loop & strategy orchestration
├── trained_results/             # Saved weights, CSV logs, plots
├── utils/                       # Reusable helpers (data, generation, training)
│   ├── datasets/
│   ├── imageGenerationSupports/
│   └── trainingStrategies/
├── visuallization/              # Notebooks for analysis / EDA / reports
├── pdb_run.py                   # Main entry-point script
├── requirements.txt             # Pinned Python deps
├── test.ipynb                   # Quick experimentation notebook
└── train-model-v4-600-60.ipynb  # Full training notebook
```

> **Rule of thumb:** each top-level folder should answer *one* question.
> `models/` → "what network?", `trainer/` → "how to train?", `evaluations/` → "how well did it do?".

---

## 2. Folder-by-Folder Specification

### 2.1 [models/](models/)

Each file = **one architecture family**. Each file exposes a single `load_<ArchName>()` factory.

```
models/
├── __init__.py            # Re-exports all loaders
├── resnet.py              # load_Resnet(name, num_classes, pretrained_path, device)
├── convnext.py            # load_ConvNeXt(...)
├── coAtNet.py             # load_CoAtNet(...)
├── efficientNetV2.py      # load_EfficientNetV2(...)
├── maxViT.py              # load_VIT_SizeT(...)
├── regNetY16GF.py         # load_RegNetY16GF(...)
└── swinV2B.py             # load_SwinV2B(...)
```

**Function signature contract (every loader must follow):**

```python
def load_<Arch>(name: str,
                num_classes: int,
                pretrained_path: str = "",
                device: str = "cpu") -> nn.Module:
    # 1) build the backbone (torchvision / timm)
    # 2) replace classifier head with nn.Linear(in_features, num_classes)
    # 3) optionally load a checkpoint from `pretrained_path`
    # 4) return the model
```

**`__init__.py` exposes everything:**

```python
from .resnet import load_Resnet
from .convnext import load_ConvNeXt
# ... add new arch here when you create it
```

**Example — adding a new model `vit_h_14`:**

```python
# models/vit_h_14.py
import torch.nn as nn
import torchvision.models as models

def load_ViT_H14(name, num_classes, pretrained_path="", device="cpu"):
    model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device)["net"])
    return model
```

---

### 2.2 [utils/](utils/)

Reusable, **stateless** helpers. Three subdomains, each in its own folder.

#### 2.2.1 [utils/datasets/](utils/datasets/)

```
utils/datasets/
└── <task>_ds.py           # e.g. pdb_ds.py
```

Each `<task>_ds.py` should contain:

| Symbol | Purpose |
|---|---|
| `LoadData(folder, class_names, isDebug, showSize)` | Walk a class-folder tree → return `(images, labels)` lists |
| `imagenet_mean`, `imagenet_std` | Normalization constants |
| `train_tf(image_size)` | torchvision augmentation pipeline for training |
| `val_tf(image_size)` | Resize + normalize for validation |
| `test_tf(image_size)` | Same as val_tf (kept separate for future divergence) |
| `class <Task>Dataset(Dataset)` | `__init__`, `__len__`, `__getitem__` switching on `train/val/test` |
| `real_<task>_testset(path, class_names)` | Load a *real-world* held-out test set |
| `get_classes(path)` | Auto-build a `{idx: name}` mapping from folder names |

**Example skeleton for a new task — `mnist_ds.py`:**

```python
class MNISTDataset(Dataset):
    def __init__(self, images, labels, image_size, type_transform="train"):
        self.images, self.labels = images, labels
        self.train_tf = train_tf(image_size)
        self.val_tf   = val_tf(image_size)
        self.test_tf  = test_tf(image_size)
        self.transform = type_transform

    def __getitem__(self, idx):
        img = {"train": self.train_tf,
               "val":   self.val_tf,
               "test":  self.test_tf}[self.transform](self.images[idx])
        return img, self.labels[idx]
```

---

#### 2.2.2 [utils/imageGenerationSupports/](utils/imageGenerationSupports/)

One-off **automation scripts** that pre-process raw data into trainable images.
These are runnable scripts, not importable modules — keep paths configurable at the top.

```
utils/imageGenerationSupports/
├── hdf2pngScriptAutomation.py        # Raw format → PNG stack
└── GeneratingProteinGIFAutomation.py # PNG stack → labeled GIF
```

**Naming convention:** `<source>2<target>ScriptAutomation.py` or `Generating<Artifact>Automation.py`.

**Each script should contain:**

```python
# 1) Configurable paths at the top
input_path  = "/path/to/raw"
output_path = "/path/to/processed"

# 2) Iterate input
for item in os.listdir(input_path):
    ...
    # 3) Transform
    ...
    # 4) Save with a deterministic name (e.g. f"{idx:03d}.png")
    im.save(os.path.join(output_path, f"{idx:03d}.png"))

print("Finish!")
```

---

#### 2.2.3 [utils/trainingStrategies/](utils/trainingStrategies/)

The **how-to-train** layer. Three orthogonal concerns, three files:

```
utils/trainingStrategies/
├── freezingControl.py             # Which layers train?
├── specificOptimizerPerModel.py   # Which optimizer + param groups?
└── specificLRSchedulerPerModel.py # How does LR change over time?
```

**`freezingControl.py` — required functions:**

```python
def freeze_backbone(model, keep_head=True): ...
def unfreeze_last_n_stages(model, n=2):     ...   # supports ResNet, ConvNeXt, Swin, MaxViT, CoAtNet, RegNet, EfficientNet
def unfreeze_all(model):                    ...
```

When adding a new architecture: extend the `if hasattr(model, ...)` ladder inside `unfreeze_last_n_stages`.

**`specificOptimizerPerModel.py` — single dispatch function:**

```python
def specificOptimizerPerModel(modelName, model, learning_rate):
    if "<Transformer>" in modelName:   # AdamW + no-WD param groups for norm/bias/pos_embed
        return torch.optim.AdamW(...)
    elif "<CNN-style>" in modelName:   # plain AdamW
        return torch.optim.AdamW(...)
    elif modelName in ["<Heavy-CNN>"]: # SGD + momentum + nesterov
        return torch.optim.SGD(...)
    raise ValueError(f"Model {modelName} is not supported.")
```

**`specificLRSchedulerPerModel.py` — schedule factories:**

```python
def cosine_warmup_schedule(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
```

Add other factories (`onecycle_schedule`, `step_schedule`, …) as new functions in this same file.

---

### 2.3 [trainer/](trainer/)

The **orchestrator** — one class per task. Wires `models/` + `utils/trainingStrategies/` + `evaluations/`.

```
trainer/
└── <Task>_Trainer.py       # e.g. PDB42_Trainer.py
```

**Required class shape:**

```python
class <Task>_Trainer:
    def __init__(self, model, device, configs,
                 class_names=None, topk=(1,3,5,10,20),
                 start_epoch=1, label_smoothing=0.0,
                 real_images_per_class=None, real_labels_per_class=None):
        # 1) move model to device
        # 2) init optimizer    via specificOptimizerPerModel(...)
        # 3) init scheduler    via cosine_warmup_schedule(...)
        # 4) init loss_fn      = nn.CrossEntropyLoss(label_smoothing=...)
        # 5) open tracking_csv and write header
        # 6) zero-out tracking lists (train_loss_list, val_acc_list, lr_list, ...)

    def train_one_epoch(self, loader): ...
    def validate(self, loader):        ...
    def topk_accuracy(self, logits, targets, ks): ...
    def save_checkpoint(self, path):   ...
    def run(self, train_loader, val_loader, num_epochs): ...
```

**CSV header convention (keep this stable across projects so plotting scripts reuse):**

```
epoch, train_acc, train_loss, val_acc, val_loss,
topk1train_acc, topk3train_acc, topk5train_acc, topk10train_acc, topk20train_acc,
topk1val_acc,   topk3val_acc,   topk5val_acc,   topk10val_acc,   topk20val_acc,
learning_rate
```

---

### 2.4 [evaluations/](evaluations/)

Plotting + metric helpers that *read* the CSV produced by the trainer.

```
evaluations/
└── evaluation_<task>.py    # e.g. evaluation_pdb.py
```

**Required functions:**

| Function | Purpose |
|---|---|
| `line_chart_k_acc(tracking_csv, path2save, type="train")` | Plot top-1/3/5/10/20 accuracy curves |
| `line_chart(tracking_csv, path2save, type="train")`       | Plot loss / acc / LR curve |
| `realTest_cm(model, images, labels, class_names, save_path)` | Confusion matrix + classification report on real test set |

Keep all `matplotlib` figure I/O here — do **not** plot from inside the trainer.

---

### 2.5 [trained_results/](trained_results/)

**Output-only**, never edited by hand. One subfolder per experiment:

```
trained_results/
└── <model>_<dataset>_<date>/
    ├── best.pth
    ├── tracking.csv
    ├── train_topk_accuracy_line_chart.png
    ├── val_topk_accuracy_line_chart.png
    └── confusion_matrix.png
```

Add this folder to `.gitignore` once it grows past a few MB.

---

### 2.6 [visuallization/](visuallization/)

Exploratory Jupyter notebooks. Keep code *thin* — import from `utils/` and `evaluations/` rather than re-defining helpers.

```
visuallization/
├── EDA_<task>.ipynb              # Dataset stats, class balance, sample grids
├── DoubleCheckResult.ipynb       # Sanity-check predictions on a few samples
├── combineResult.ipynb           # Merge CSVs across runs into one chart
├── comparisonModelResult.ipynb   # Side-by-side bar charts of multiple models
└── show<Task>Report.ipynb        # Final report-ready figures
```

---

### 2.7 [3D_PDB_Dataset/](3D_PDB_Dataset/) (or `dataset/`)

Class-folder layout consumed by `LoadData`:

```
<DatasetName>/
├── <split_name>/                # e.g. 90_12
│   ├── TrainProteinPNG90/
│   │   ├── CLASS_A/
│   │   │   ├── sample_001.png
│   │   │   └── ...
│   │   ├── CLASS_B/
│   │   └── ...
│   ├── ValProteinPNG90/
│   └── TestProteinPNG90/
```

The folder name *is* the class label — `get_classes()` builds `{idx: folder_name}`.

---

## 3. Root Files

| File | Purpose |
|---|---|
| `pdb_run.py` | CLI entry: parses args/configs, instantiates `<Task>_Trainer`, calls `.run()` |
| `requirements.txt` | Pinned versions: `torch`, `torchvision`, `timm`, `opencv-python`, `h5py`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`, `Pillow` |
| `train-*.ipynb` | Full training notebook with markdown context — good for sharing |
| `test.ipynb` | Scratchpad — never depend on it from importable code |

**`pdb_run.py` skeleton:**

```python
from models import load_Resnet, load_SwinV2B, ...  # noqa
from utils.datasets.pdb_ds import (LoadData, PBD42Dataset, get_classes,
                                   real_protein_testset)
from trainer.PDB42_Trainer import PDB42_Trainer

configs = {
    "model": "Resnet50",
    "lr": 3e-4,
    "max_epoch_num": 60,
    "batch_size": 32,
    "image_size": 224,
    "tracking_csv": "trained_results/resnet50_pdb_2026-05-12/tracking.csv",
    ...
}

class_names = get_classes(configs["train_path"])
train_imgs, train_lbls = LoadData(configs["train_path"], class_names)
val_imgs,   val_lbls   = LoadData(configs["val_path"],   class_names)

train_ds = PBD42Dataset(train_imgs, train_lbls, configs["image_size"], "train")
val_ds   = PBD42Dataset(val_imgs,   val_lbls,   configs["image_size"], "val")

model = load_Resnet("Resnet50", num_classes=len(class_names))
trainer = PDB42_Trainer(model, "cuda", configs, class_names=class_names)
trainer.run(train_ds, val_ds)
```

---

## 4. How to Start a New Project From This Template

1. Copy the top-level skeleton (empty folders + `__init__.py` files).
2. Drop in **one** model loader in `models/` → exercise the contract.
3. Implement `utils/datasets/<task>_ds.py` with the seven required symbols.
4. Implement the three `utils/trainingStrategies/` files (or copy & adapt).
5. Write `trainer/<Task>_Trainer.py` — wire everything together.
6. Write `evaluations/evaluation_<task>.py` for plots.
7. Stand up `pdb_run.py`-style entry script.
8. Add notebooks under `visuallization/` *last*, once metrics are flowing.

---

## 5. Naming Conventions Cheat Sheet

| Concept | Pattern | Example |
|---|---|---|
| Model loader file | `<arch>.py` (camelCase ok) | `swinV2B.py` |
| Model loader fn | `load_<Arch>` | `load_SwinV2B` |
| Dataset module | `<task>_ds.py` | `pdb_ds.py` |
| Dataset class | `<Task>Dataset` | `PBD42Dataset` |
| Trainer class | `<Task>_Trainer` | `PDB42_Trainer` |
| Evaluation module | `evaluation_<task>.py` | `evaluation_pdb.py` |
| Generation script | `<src>2<dst>ScriptAutomation.py` | `hdf2pngScriptAutomation.py` |
| Result folder | `<model>_<dataset>_<YYYY-MM-DD>/` | `resnet50_pdb_2026-05-12/` |
