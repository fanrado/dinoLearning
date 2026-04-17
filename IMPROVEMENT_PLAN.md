# Improvement Plan — DINO on PetImages (Cats/Dogs, 25k)

## Context

Repo trains DINO (ViT-S/16) self-supervised on PetImages (12.5k Cat + 12.5k Dog). Current two-pass iterative strategy (Pass 1: full aug / shape; Pass 2: minimal aug / color) has been run. Observed result:

- Pass 1 k-NN (5/10/20-NN): **87.82 / 88.66 / 88.96**
- Pass 2 k-NN (5/10/20-NN): **85.92 / 86.88 / 87.54** → **Pass 2 degrades quality**

Constraint: no additional data. Task = improve k-NN / linear accuracy using only the existing 25k images.

The plan below is ordered by expected impact vs effort. Each step is standalone and measurable against the current 88.96 (20-NN) baseline.

---

## Root causes (why current setup underperforms)

1. **Pass 2 hurts, not helps.** Running DINO self-sup again with minimal aug on a dataset with reliable color is just reducing the effective augmentation diversity → teacher collapses toward easier solution → features regress. Pass 2 as self-sup is mis-designed.
2. **Pass 1 LR is too low.** Script uses `P1_LR=0.000125` (linearly scaled from 0.0005 base @ batch 256). For a small dataset + long 200 epochs, this under-trains. Loss flat at ~5.0 from epoch ~100 onward (centering collapse risk).
3. **Dataset has known corrupt files.** PetImages ships with broken / non-image files (well known issue). `Truncated File Read` warning in logs confirms. Silent skips → label noise.
4. **ViT-S/16 on 25k is capacity-mismatched.** Small dataset benefits from smaller patch (more tokens, stronger inductive load) or smaller arch, not blindly ViT-S/16.
5. **No early-stop / best-ckpt tracking.** Latest ckpt used for k-NN; no periodic eval during training.
6. **Evaluation weak.** k-NN only; no linear probe result committed, no confusion matrix, no per-class.

---

## Plan — step by step

### Step 1 — Clean dataset (cheap, mandatory)
- Scan `PetImages/Cat` and `PetImages/Dog`; drop files that:
  - fail `PIL.Image.open().verify()`
  - are 0 bytes
  - are not JPG/PNG/GIF
- Expected removal ~1–2% (well documented for PetImages).
- **Why:** removes label noise before any re-training. Single biggest "free" lift.

### Step 2 — Drop Pass 2 DINO, replace with supervised linear/fine-tune head
- Current Pass 2 (self-sup with minimal aug) empirically worsens features → remove.
- Use Pass 1 teacher checkpoint as frozen backbone.
- Run `eval_linear.py` (already in repo) for the supervised head → gives real accuracy number AND reuses the shape-pretrained backbone.
- Optional: full fine-tune with very low LR (1e-5) for last 1–2 transformer blocks only.
- **Why:** matches what README *says* Pass 2 should be (supervised) but is not in `run_dino_iterative.sh`. Script currently runs `main_dino.py` twice.

### Step 3 — Fix Pass 1 hyperparameters
- LR: raise to `5e-4` effective (batch-size-aware): use `lr = 5e-4 * batch/256`, and **increase batch via gradient accumulation** to at least 128 → target effective batch 128, LR = 2.5e-4.
- `teacher_temp`: try **0.07** with `warmup_teacher_temp_epochs=30` (per README "boosting" recipe) instead of flat 0.04.
- `momentum_teacher`: keep 0.996 but set schedule → 1.0 via cosine (already in utils).
- `out_dim`: try **4096** (not 65536). 65536 is for ImageNet-scale; on 25k it overparameterizes the head.
- Epochs: 200 is fine; add **periodic k-NN eval every 20 epochs** to pick best ckpt, not last.
- **Why:** default/scaled settings are ImageNet-tuned; here small dataset + long epochs already trips teacher collapse symptoms (loss floor ~5.0).

### Step 4 — Smaller patch → more tokens (same params)
- Switch `--patch_size 16` → `--patch_size 8` (ViT-S/8, same 21M params).
- Doubles token count, quadruples attention compute, but on 224×224 still fits a single GPU at batch 32 with fp16.
- DINO paper: ViT-S/8 beats ViT-S/16 on k-NN by ~4 points.
- **Why:** stronger spatial localization matters on small datasets where inductive bias compensates for limited data.

### Step 5 — Better augmentation calibration (not "more")
- Pass 1 uses default ColorJitter (bright 0.4, contrast 0.4, sat 0.2, hue 0.1). For cats vs dogs, **hue is a weak discriminator** (tabby ↔ tabby-dog) so aggressive hue is fine; but:
  - Add **RandAugment(n=2, m=9)** after crop — cheaply adds geometric diversity.
  - Add **CutMix/MixUp** in Pass 2 (linear/fine-tune only; not compatible with DINO loss in Pass 1).
- **Why:** with fixed data, aug diversity is the lever. RandAugment gives unseen-view coverage without changing semantics.

### Step 6 — Self-distillation refinement loop (replaces broken Pass 2)
Instead of running DINO twice, distill backbone into itself using pseudo-labels:
1. Freeze Pass 1 teacher.
2. Extract features → run k-means(k=2) on train split.
3. Use cluster assignment as **pseudo-label**; train supervised head + last block.
4. Use confidence threshold (>0.9) to mine "easy" samples → re-run on confident subset.
- **Why:** gives structured supervision signal derived from the existing self-sup features, without manual labels and without a second self-sup pass that collapses.

### Step 7 — Evaluation upgrades
- Add to eval script:
  - **Linear probe top-1** (already supported via `eval_linear.py`; just commit the run).
  - **Confusion matrix** + per-class accuracy → catch asymmetric Cat-vs-Dog confusion.
  - **k-NN with multiple k** (5/10/20 already there) + **cosine-vs-euclidean** comparison.
  - **Feature alignment/uniformity** metrics (Wang & Isola 2020) as collapse detectors.
- **Why:** single scalar k-NN hides regressions; Pass 2 degradation would have been caught earlier.

### Step 8 — Stabilize training infra
- Save checkpoint named `best_knn.pth` (highest k-NN every eval), not only `checkpoint.pth`.
- Add `--resume` path sanity (already exists in main_dino).
- Log gradient norms + teacher/student cosine → early collapse detection.
- **Why:** cheap reliability win.

---

## Files to modify (critical paths)

- `run_dino_iterative.sh` — replace Pass 2 block with `eval_linear.py` call; raise Pass 1 LR; add patch_size=8 option.
- `main_dino.py` — add RandAugment option in `DataAugmentationDINO`; add periodic k-NN eval hook (every N epochs); save `best_knn.pth`.
- `eval_knn.py` — add confusion matrix, cosine vs euclidean switch (already near [eval_knn.py:30-65](eval_knn.py#L30-L65)).
- `eval_linear.py` — ensure minimal-aug pipeline matches Pass 2 intent (already close; verify).
- **New**: `clean_petimages.py` — one-shot dataset sanitization (Step 1).
- **New**: `pseudo_label_distill.py` — Step 6.

Reuse (don't reinvent):
- [utils.py](utils.py) → already has `cosine_scheduler`, `bool_flag`, `GaussianBlur`, `Solarization`, `load_pretrained_weights`.
- [eval_linear.py](eval_linear.py) → already has frozen-backbone linear probe.
- [plot_features.py](plot_features.py) → already produces t-SNE / PCA.
- `sklearn.cluster.KMeans` for Step 6 (no new dep).

---

## Verification

1. Run Step 1 cleaning → log number of dropped files; rerun Pass 1 20-NN → expect same or +0.5 accuracy baseline.
2. After Step 3 hyperparam change → Pass 1 20-NN should be ≥ 89.5 (from 88.96).
3. After Step 4 (patch 8) → Pass 1 20-NN should be ≥ 91.
4. After Step 2 (linear probe replaces broken Pass 2) → linear top-1 should be ≥ 93.
5. After Step 6 (self-distill) → linear top-1 target ≥ 94.5.
6. Regression guard: every run writes `features/knn_results.log` + new `confusion_matrix.png` + `linear_results.log`; compare against baseline file `OUTPUT_DINO_ITERATIVE/pass1_shape/features/knn_results.log` (current best 88.96 @ 20-NN).

---

## Order of execution (suggested)

1. Step 1 (clean data) — ~10 min
2. Step 7 + Step 8 (eval + best-ckpt tracking) — infra, do before any retrain
3. Step 3 (hyperparam fix) — retrain Pass 1
4. Step 2 (linear probe) — real accuracy number
5. Step 4 (patch 8) — retrain, expensive
6. Step 5 (aug calibration) — retrain
7. Step 6 (self-distill) — last, most experimental
