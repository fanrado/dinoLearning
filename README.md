:new: *Please check out our more recent [DINOv2](https://github.com/facebookresearch/dinov2) effort in the same line of work.*

# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO. For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] [[`arXiv`](https://arxiv.org/abs/2104.14294)] [[`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)]

<div align="center">
  <img width="100%" alt="DINO illustration" src=".github/dino.gif">
</div>

---

## Iterative Training Plan: Cats & Dogs (25k samples)

This section describes a two-pass iterative training strategy applied to a 25k-sample cats/dogs dataset. The approach is designed to be **dataset-agnostic** and reusable for any binary or multi-class image dataset.

### Motivation

Visual recognition requires two complementary types of knowledge:
- **Shape/structure** — the silhouette, pose, proportions, and edges that define an object's geometry.
- **Color/texture/fine detail** — the subtle tonal patterns, fur textures, and color distributions that further discriminate classes.

Standard end-to-end supervised training mixes both signals simultaneously, often causing the model to shortcut on color rather than building robust shape representations first. The iterative strategy below forces a clean separation: the first pass builds shape-invariant representations; the second pass refines them with color and detail.

---

### Dataset Preparation

The dataset must be organized in `torchvision.datasets.ImageFolder` format, with one sub-directory per class directly under the root:

```
PetImages/
  Cat/
    cat_00001.jpg
    cat_00002.jpg
    ...
  Dog/
    dog_00001.jpg
    dog_00002.jpg
    ...
```

**No manual train/val split is required.** `eval_knn.py` performs a stratified 80/20 split internally using `sklearn.model_selection.train_test_split` (controlled by `--val_split 0.2`). `main_dino.py` uses the full dataset for self-supervised pre-training — labels are not used during Pass 1. Ensure roughly equal class balance (≈12,500 cats, ≈12,500 dogs) to avoid bias.

---

### Pass 1 — Self-Supervised Pre-training: Shape Learning

**Goal:** Train the backbone using the DINO self-supervised objective with **all available augmentations** enabled. The heavy color distortions and grayscale conversion remove color as a reliable cue, forcing the model to learn geometry, edges, and structural patterns.

#### Augmentation pipeline (DataAugmentationDINO — default behavior)

| Augmentation | Applied to | Setting |
|---|---|---|
| RandomResizedCrop | Global views | scale `[0.4, 1.0]`, size 224×224 |
| RandomResizedCrop | Local views (×8) | scale `[0.05, 0.4]`, size 96×96 |
| RandomHorizontalFlip | All views | p=0.5 |
| ColorJitter | All views | brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8 |
| RandomGrayscale | All views | **p=0.2** — key for shape invariance |
| GaussianBlur | Global view 1 | **p=1.0** (always applied) |
| GaussianBlur | Global view 2 | p=0.1 |
| GaussianBlur | Local views | p=0.5 |
| Solarization | Global view 2 | **p=0.2** — additional color disruption |
| Normalize | All views | mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) |

The combination of ColorJitter + RandomGrayscale + Solarization makes color an **unreliable** signal across the student/teacher views. The DINO loss then rewards consistency on shape features only.

#### Training command (single GPU)

```bash
python main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 200 \
    --warmup_epochs 10 \
    --batch_size_per_gpu 64 \
    --lr 0.00025 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --momentum_teacher 0.996 \
    --norm_last_layer false \
    --global_crops_scale 0.4 1.0 \
    --local_crops_scale 0.05 0.4 \
    --local_crops_number 8 \
    --data_path /path/to/cats_dogs \
    --output_dir /path/to/outputs/pass1
```

> **Note on learning rate:** The base LR of 0.0005 assumes a batch size of 256. For a batch of 64 (single GPU), the effective LR is scaled linearly: `0.0005 × 64/256 = 0.000125`. Adjust if using multiple GPUs.
>
> **Note on dataset path:** Pass `--data_path` pointing to the root of the ImageFolder (e.g. `PetImages/` containing `Cat/` and `Dog/`). `main_dino.py` loads the full dataset directly; `eval_knn.py` handles the train/val split internally via `sklearn.train_test_split`.

#### Training command (multi-GPU, recommended)

```bash
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 200 \
    --warmup_epochs 10 \
    --batch_size_per_gpu 64 \
    --lr 0.0005 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --momentum_teacher 0.996 \
    --norm_last_layer false \
    --global_crops_scale 0.4 1.0 \
    --local_crops_scale 0.05 0.4 \
    --local_crops_number 8 \
    --data_path /path/to/cats_dogs \
    --output_dir /path/to/outputs/pass1
```

#### Key hyperparameter rationale

| Parameter | Value | Why |
|---|---|---|
| `--arch vit_small --patch_size 16` | ViT-S/16 | Best compute/accuracy trade-off for ~25k images; 21M parameters, generalizes well at this scale |
| `--epochs 200` | 200 | With only 25k images, more epochs are needed to see sufficient augmented views per sample compared to ImageNet-scale training |
| `--norm_last_layer false` | disabled | Empirically improves DINO quality on smaller datasets (per official boosting recipe) |
| `--warmup_teacher_temp_epochs 30` | 30 | Stable early training; prevents teacher collapsing when dataset is small |
| `--local_crops_number 8` | 8 | 10 total views (2 global + 8 local) per image; more views improve self-supervised signal quality |
| `--global_crops_scale 0.4 1.0` | full range | Large crops anchor the global view; small min-scale exposes partial views for context learning |

#### Convergence monitoring

Track the following metrics during Pass 1 (available in training logs):

- **DINO loss**: should decrease steadily and plateau. For 25k images, expect meaningful convergence by epoch 80–100.
- **k-NN accuracy** (run periodically with `eval_knn.py`): use as a proxy for representation quality without committing to a full linear evaluation.
- **Self-attention maps** (run `visualize_attention.py`): at convergence the [CLS] attention heads should segment foreground objects (cat/dog bodies) cleanly, with no attention on background.

```bash
# Quick k-NN check after Pass 1
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
    --pretrained_weights /path/to/outputs/pass1/checkpoint.pth \
    --checkpoint_key teacher \
    --data_path /path/to/cats_dogs \
    --nb_knn 20
```

**Pass 1 is complete** when the k-NN accuracy plateaus across 10+ consecutive epochs and attention maps cleanly separate cat/dog foregrounds from backgrounds.

---

### Pass 2 — Supervised Fine-tuning: Color and Detail Learning

**Goal:** Using the shape-rich backbone from Pass 1 as initialization, fine-tune the full model (or a linear head) with **minimal augmentations**. With color information now preserved and reliable, the model learns the color distributions, fur textures, and fine-grained visual cues that further separate cats from dogs.

#### Why minimal augmentations here

In Pass 1, color augmentations were essential to block color shortcuts and build shape invariance. In Pass 2, that invariance is already baked into the weights. Applying the same aggressive augmentations would **erase the color signal** the model needs to learn in this pass. The goal is the opposite: preserve colors faithfully so the model can learn from them.

#### Augmentation pipeline (Pass 2 — minimal)

| Augmentation | Setting | Rationale |
|---|---|---|
| RandomResizedCrop(224) | scale `[0.8, 1.0]` | Gentle crop only; preserves spatial color context |
| RandomHorizontalFlip | p=0.5 | Spatial only; does not affect color |
| **No** ColorJitter | disabled | Color must be preserved as a learning signal |
| **No** RandomGrayscale | disabled | Color channels are informative |
| **No** GaussianBlur | disabled | Preserves fine texture detail |
| **No** Solarization | disabled | No color inversion |
| Normalize | mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) | Standard normalization only |

#### Option A — Linear probe (fastest, backbone frozen)

Best for quickly validating Pass 1 quality and when the backbone is already very strong.

```bash
python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
    --arch vit_small \
    --patch_size 16 \
    --pretrained_weights /path/to/outputs/pass1/checkpoint.pth \
    --checkpoint_key teacher \
    --epochs 100 \
    --lr 0.001 \
    --batch_size_per_gpu 128 \
    --data_path /path/to/cats_dogs \
    --output_dir /path/to/outputs/pass2_linear
```

The linear probe freezes all backbone weights and only trains the final classification layer. The minimal augmentation constraint is naturally enforced via the `eval_linear.py` transform pipeline (RandomResizedCrop + HorizontalFlip only, no color distortion).

#### Option B — Full fine-tuning (deeper color integration)

Unfreezes the backbone and trains end-to-end at a very low learning rate. This lets the backbone itself adapt to color cues while retaining the shape structure from Pass 1. This is the recommended path for maximizing final accuracy.

```bash
python main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 100 \
    --warmup_epochs 5 \
    --batch_size_per_gpu 64 \
    --lr 0.00005 \
    --min_lr 1e-7 \
    --weight_decay 0.04 \
    --weight_decay_end 0.04 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 0 \
    --momentum_teacher 0.9996 \
    --norm_last_layer true \
    --global_crops_scale 0.8 1.0 \
    --local_crops_scale 0.5 0.8 \
    --local_crops_number 2 \
    --pretrained_weights /path/to/outputs/pass1/checkpoint.pth \
    --data_path /path/to/cats_dogs \
    --output_dir /path/to/outputs/pass2_finetune
```

> **Key changes from Pass 1:**
> - LR reduced 10× (`0.00005` vs `0.0005`) — prevents catastrophic forgetting of shape features
> - `--global_crops_scale 0.8 1.0` — conservative crops, preserves color context
> - `--local_crops_number 2` — fewer local crops; less distortion per image
> - `--local_crops_scale 0.5 0.8` — larger local crop scale, less extreme viewpoint change
> - No ColorJitter / Grayscale / Solarization applied (set `--color_jitter 0.0` if exposed as a flag, otherwise modify `DataAugmentationDINO` in `main_dino.py` directly)

#### Modifying DataAugmentationDINO for Pass 2

To fully disable color augmentations, edit `main_dino.py` in the `DataAugmentationDINO.__init__` method. Replace the color transform block with identity transforms:

```python
# Pass 2: minimal augmentation — comment out or replace the color block
color_jitter = transforms.Compose([])          # no-op: was ColorJitter
# transforms.RandomGrayscale(p=0.2)            # disabled
# GaussianBlur(...)                             # disabled
# Solarization(...)                             # disabled
```

This ensures the augmentation pipeline for Pass 2 is strictly:
```
RandomResizedCrop(scale=[0.8, 1.0]) → RandomHorizontalFlip → Normalize
```

#### Convergence monitoring for Pass 2

- **Linear / full fine-tuning accuracy on val set**: primary convergence signal. With 25k images and a strong Pass 1 backbone, expect >90% binary accuracy within 30–50 epochs.
- **Loss curve**: should decrease quickly (good initialization from Pass 1) and stabilize.
- **Attention maps**: Run `visualize_attention.py` again. Attention should now be drawn to color-discriminative regions (orange fur patches, eye color, coat patterns) in addition to shape boundaries.

---

### Evaluation After Each Pass

```bash
# k-NN evaluation (no training, pure feature quality check)
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
    --pretrained_weights /path/to/outputs/passN/checkpoint.pth \
    --checkpoint_key teacher \
    --data_path /path/to/cats_dogs \
    --nb_knn 5 10 20

# Linear probe evaluation
python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
    --arch vit_small \
    --patch_size 16 \
    --pretrained_weights /path/to/outputs/passN/checkpoint.pth \
    --checkpoint_key teacher \
    --epochs 100 \
    --data_path /path/to/cats_dogs \
    --output_dir /path/to/outputs/passN_linear_eval
```

Compare k-NN and linear accuracy across Pass 1 and Pass 2 to confirm each pass adds value.

---

### Summary: Two-Pass Iterative Training

| | Pass 1 (Pre-training) | Pass 2 (Fine-tuning) |
|---|---|---|
| **Script** | `main_dino.py` | `eval_linear.py` or `main_dino.py` |
| **Supervision** | Self-supervised (DINO) | Supervised (labels used) |
| **Augmentations** | Full: ColorJitter, Grayscale, Blur, Solarize, multi-crop | Minimal: Crop + Flip only |
| **Color signal** | Disrupted (shape forced) | Preserved (color learned) |
| **LR** | 0.0005 | 0.001 (linear) / 0.00005 (full FT) |
| **Epochs** | 200 | 100 |
| **Crop scale** | Global: 0.4–1.0, Local: 0.05–0.4 | Global: 0.8–1.0, Local: 0.5–0.8 |
| **Primary signal learned** | Shape, edges, structure | Color, texture, fine detail |
| **Output** | Pre-trained backbone (teacher) | Classifier checkpoint |

---

### Extending to Other Datasets

This two-pass pattern is intentionally dataset-agnostic. To apply it to a new dataset:

1. **Organize data** in `ImageFolder` format under a new directory (e.g., `data/birds/train/{class}/`).
2. **Run Pass 1** with the same command, replacing `--data_path`. No label information is used — the self-supervised objective works on any image collection.
3. **Run Pass 2** pointing `--pretrained_weights` to the Pass 1 checkpoint. If the new dataset has more classes, adjust the classifier head (handled automatically in `eval_linear.py` via `--num_labels`).
4. **Scale epochs** proportionally to dataset size. A rough guide: aim for ~1,000–2,000 total augmented-view passes per sample. With 25k images and 200 epochs × 10 views, that is `25,000 × 200 × 10 = 50M` total crop observations.

The key invariant to preserve across datasets: **Pass 1 must destroy color; Pass 2 must preserve it.**

---

## Pretrained models
You can choose to download only the weights of the pretrained backbone used for downstream tasks, or the full checkpoint which contains backbone and projection head weights for both student and teacher networks. We also provide the backbone in `onnx` format, as well as detailed arguments and training/evaluation logs. Note that `DeiT-S` and `ViT-S` names refer exactly to the same architecture.

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>74.5%</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deits16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>21M</td>
    <td>78.3%</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deits8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>76.1%</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitb16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>85M</td>
    <td>77.4%</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitb8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>67.5%</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
</table>

We also release XCiT models ([[`arXiv`](https://arxiv.org/abs/2106.09681)] [[`code`](https://github.com/facebookresearch/xcit)]) trained with DINO:
<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="5">download</th>
  </tr>
  <tr>
    <td>xcit_small_12_p16</td>
    <td>26M</td>
    <td>76.0%</td>
    <td>77.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p8</td>
    <td>26M</td>
    <td>77.1%</td>
    <td>79.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p16</td>
    <td>84M</td>
    <td>76.4%</td>
    <td>78.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p8</td>
    <td>84M</td>
    <td>77.9%</td>
    <td>80.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
</table>

### Pretrained models on PyTorch Hub
```python
import torch
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
xcit_small_12_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16')
xcit_small_12_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8')
xcit_medium_24_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16')
xcit_medium_24_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
```

## Training

### Documentation
Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). For a glimpse at the full documentation of DINO training please run:
```
python main_dino.py --help
```

### Vanilla DINO training :sauropod:
Run DINO with ViT-small network on a single node with 8 GPUs for 100 epochs with the following command. Training time is 1.75 day and the resulting checkpoint should reach 69.3% on k-NN eval and 74.0% on linear eval. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

### Multi-node training
We use Slurm and [submitit](https://github.com/facebookincubator/submitit) (`pip install submitit`). To train on 2 nodes with 8 GPUs each (total 16 GPUs):
```
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

<details>
<summary>
DINO with ViT-base network.
</summary>

```
python run_with_submitit.py --nodes 2 --ngpus 8 --use_volta32 --arch vit_base  --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

### Boosting DINO performance :t-rex:
You can improve the performance of the vanilla run by:
- training for more epochs: `--epochs 300`,
- increasing the teacher temperature: `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`.
- removing last layer normalization (only safe with `--arch vit_small`): `--norm_last_layer false`,

<details>
<summary>
Full command.
</summary>

```
python run_with_submitit.py --arch vit_small --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

The resulting pretrained model should reach 73.3% on k-NN eval and 76.0% on linear eval. Training time is 2.6 days with 16 GPUs. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.

### ResNet-50 and other convnets trainings
This code also works for training DINO on convolutional networks, like ResNet-50 for example. We highly recommend to adapt some optimization arguments in this case. For example following is a command to train DINO on ResNet-50 on a single node with 8 GPUs for 100 epochs. We provide [training logs](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_log.txt) and [final checkpoint](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_checkpoint.pth) for this run.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

## Self-attention visualization
You can look at the self-attention of the [CLS] token on the different heads of the last layer by running:
```
python visualize_attention.py
```

<div align="center">
  <img width="100%" alt="Self-attention from a Vision Transformer with 8x8 patches trained with DINO" src=".github/attention_maps.png">
</div>

## Self-attention video generation
You can generate videos like the one on the blog post with `video_generation.py`.

https://user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4

Extract frames from input video and generate attention video:
```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path input/video.mp4 \
    --output_path output/ \
    --fps 25
```

Use folder of frames already extracted and generate attention video:
```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path output/frames/ \
    --output_path output/ \
    --resize 256 \
```

Only generate video from folder of attention maps images:
```
python video_generation.py --input_path output/attention \
    --output_path output/ \
    --video_only \
    --video_format avi
```


## Evaluation: k-NN classification on ImageNet
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, run:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --data_path /path/to/imagenet
```
If you choose not to specify `--pretrained_weights`, then DINO reference weights are used by default. If you want instead to evaluate checkpoints from a run of your own, you can run for example:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/imagenet 
```

## Evaluation: Linear classification on ImageNet
To train a supervised linear classifier on frozen weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /path/to/imagenet
```

We release the logs and weights from evaluating the different models:

<table>
  <tr>
    <th>arch</th>
    <th>top-1 ImageNet</th>
    <th colspan="2">linear evaluation</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p16</td>
    <td>77.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p8</td>
    <td>79.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p16</td>
    <td>78.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p8</td>
    <td>80.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
</table>

You can check the performance of the pretrained weights on ImageNet validation set by running the following command lines:
```
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch vit_small --patch_size 8 --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch vit_base --patch_size 16 --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch vit_base --patch_size 8 --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch resnet50 --data_path /path/to/imagenet/train
```

## Evaluation: DAVIS 2017 Video object segmentation
Please verify that you're using pytorch version 1.7.1 since we are not able to reproduce the results with most recent pytorch 1.8.1 at the moment.

**Step 1: Prepare DAVIS 2017 data**  
```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
```

**Step 2: Video object segmentation**  
```
python eval_video_segmentation.py --data_path $HOME/davis-2017/DAVIS/ --output_dir /path/to/saving_dir
```

**Step 3: Evaluate the obtained segmentation**  
```
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
python $HOME/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /path/to/saving_dir --davis_path $HOME/davis-2017/DAVIS/
```

## Evaluation: Image Retrieval on revisited Oxford and Paris
Step 1: Prepare revisited Oxford and Paris by following [this repo](https://github.com/filipradenovic/revisitop).

Step 2: Image retrieval (if you do not specify weights with `--pretrained_weights` then by default [DINO weights pretrained on Google Landmark v2 dataset](https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth) will be used).

Paris:
```
python -m torch.distributed.launch --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 512 --multiscale 1 --data_path /path/to/revisited_paris_oxford/ --dataset rparis6k
```

Oxford:
```
python -m torch.distributed.launch --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path /path/to/revisited_paris_oxford/ --dataset roxford5k
```

## Evaluation: Copy detection on Copydays
Step 1: Prepare [Copydays dataset](https://lear.inrialpes.fr/~jegou/data.php#copydays).

Step 2 (opt): Prepare a set of image distractors and a set of images on which to learn the whitening operator.
In our paper, we use 10k random images from YFCC100M as distractors and 20k random images from YFCC100M (different from the distractors) for computing the whitening operation.

Step 3: Run copy detection:
```
python -m torch.distributed.launch --use_env --nproc_per_node=1 eval_copy_detection.py --data_path /path/to/copydays/ --whitening_path /path/to/whitening_data/ --distractors_path /path/to/distractors/
```
We report result on the strong subset. For example in the stdout from the command above we get: `eval on strong mAP=0.858`.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
