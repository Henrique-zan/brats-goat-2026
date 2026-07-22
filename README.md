# Generalizable Brain Tumor Segmentation with Self-Training and Tumor-Aware Deformations

Code and experiments for our submission to the **Generalizability Across Tumors
(BraTS-GoAT)** task of the **BraTS 2026 Challenge**.

Challenge team: **Team_hen** — [synapse.org/Team:3587525](https://www.synapse.org/Team:3587525)

The approach extends **nnU-Net v2** (3D full-resolution, Large Residual Encoder preset)
with two components:

1. **Self-training** — a modified nnU-Net *teacher* generates pseudo-labels for the
   unlabeled training pool; the most confident 75% are merged with the labeled data.
2. **Tumor-aware deformable augmentation** — elastic/scaling deformations confined to
   the lesion and a smooth surrounding halo, so tumor shape and size vary while the
   rest of the anatomy is preserved.

## Repository layout

```
.
├── docker/                  # containerized submission (Synapse / challenge evaluation)
│   ├── Dockerfile           # base + nnU-Net 2.6.4 + custom trainer + weights
│   ├── run_model.py         # entrypoint: /input -> flat predictions in /output
│   ├── nnunet_trainer/      # tumor-aware deformable augmentation (the method itself)
│   └── model/               # trained weights go here (not tracked — see its README)
├── notebooks/
│   ├── 01_teacher/          # teacher training + pseudo-label generation
│   ├── 02_students/         # the four student configurations (paper Table 3)
│   └── 03_retention_sweep/  # pseudo-label retention study (paper Table 1)
├── configs/                 # nnU-Net plans, dataset definitions and splits actually used
│   ├── plans/               # resolved nnUNetResEncUNetLPlans per run
│   ├── datasets/            # dataset.json per run (labels, channels, case counts)
│   ├── splits/              # splits_final.json per run (the exact train/val partition)
│   └── environment.txt      # Python / torch / nnU-Net / GPU used for training
└── results/                 # per-case scores returned by the challenge platform
```

All notebooks are provided **as executed**, including their outputs — the full nnU-Net
training logs, epoch by epoch — so each reported configuration can be traced back to its
actual run.
Every experiment uses a distinct nnU-Net `DATASET_ID` and its own isolated
`nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` directories, so runs never
overwrite each other.

## Method components

| Component | Where |
|---|---|
| Tumor-aware deformable augmentation (paper Sec. 3.3) | `docker/nnunet_trainer/nnUNetTrainer_deformAug_agressive.py`, applied in `notebooks/02_students/nnUnet_base_1fold_ssl_deform_aug_agressive.ipynb` (submitted model) |
| Teacher: GroupNorm + axial attention + Hausdorff loss, region-based, 5-fold (Sec. 3.1) | `notebooks/01_teacher/Modified_nnUNet_submission.ipynb` |
| Pseudo-label generation + confidence ranking (Sec. 3.2) | `notebooks/01_teacher/Modified_nnUNet_pseudolabel_generator.ipynb` |
| Inference entrypoint used for the submission | `docker/run_model.py` |

The two augmentation notebooks embed the trainer source inline (they write it into the
`nnunetv2` package before training). That embedded copy is byte-identical to
`docker/nnunet_trainer/nnUNetTrainer_deformAug_agressive.py`, so the container runs the
same augmentation code that was trained with.

## Experiments → paper tables

### Table 1 — pseudo-label retention study

Used to select the retention fraction. These four runs use the standard `nnUNetTrainer`
(no tumor-aware augmentation), so they are identical to the *w/ pseudo-labels* student
**except** for the amount of pseudo-labels and for an **80%/20%** split adopted only for
this comparison. `retention_75` and `student_2_pseudo75` therefore share the same 803
pseudo-labels and differ only in the split.

| Retention | Notebook | Pseudo-labels kept | Scores |
|---|---|---|---|
| 25% | `03_retention_sweep/nnUnet_base_1fold_ssl_25.ipynb` | 268 cases (DS956) | `results/retention_25.csv` |
| 50% | `03_retention_sweep/nnUnet_base_1fold_ssl_50.ipynb` | 536 cases (DS955) | `results/retention_50.csv` |
| **75% (selected)** | `03_retention_sweep/nnUnet_base_1fold_ssl_75.ipynb` | 803 cases (DS954) | `results/retention_75.csv` |
| 100% | `03_retention_sweep/nnUnet_base_1fold_ssl_100.ipynb` | 1,071 cases (DS953) | `results/retention_100.csv` |

`results/pseudolabel_confidence.csv` holds the teacher confidence (`mean_fg_prob`) and
predicted volumes per unlabeled case — the ranking used to select the top 75%
(threshold ≈ 0.9722, 803 of 1,071 non-empty cases).

### Table 2 — augmentation parameters

The nine parameters of the tumor-aware deformable augmentation are the `_CFG` dictionary
at the top of `docker/nnunet_trainer/nnUNetTrainer_deformAug_agressive.py`: application
probability, minimum tumor size, elastic field smoothness and maximum displacement,
scale range, dilation iterations, halo smoothing, ROI margin and edge-fade width.

### Table 3 — main ablation

All four students share the same setup (Large Residual Encoder, patch 160×192×160,
batch 3, 1000 epochs, patient-level **95%/5%** split) and differ **only** in whether
pseudo-labels and/or the tumor-aware augmentation are used.

| Paper row | Notebook | What it does | Scores |
|---|---|---|---|
| Teacher — labeled data only | `01_teacher/Modified_nnUNet_submission.ipynb` | Modified nnU-Net teacher, 5-fold ensemble, 250 epochs (DS999) | `results/teacher.csv` |
| Student — labeled data only | `02_students/nnUnet_base_1fold_supervised_less_val.ipynb` | Baseline: labeled data, standard trainer, no augmentation (DS952) | `results/student_1_labeled_only.csv` |
| Student — w/ pseudo-labels | `02_students/nnUnet_base_1fold_ssl_75_less_val.ipynb` | Self-training: labeled + top-75% pseudo-labels (DS960) | `results/student_2_pseudo75.csv` |
| Student — w/ tumor-aware aug. | `02_students/nnUnet_base_1fold_labeled_deform_aug_agressive.ipynb` | Augmentation **without** pseudo-labels — isolates its effect (DS961) | `results/student_3_labeled_aug.csv` |
| **Student (submitted)** — pseudo-labels + aug. | `02_students/nnUnet_base_1fold_ssl_deform_aug_agressive.ipynb` | Full method: top-75% pseudo-labels + tumor-aware augmentation (DS959) | `results/student_4_submitted.csv` |

## Configurations

`configs/` holds the files nnU-Net actually resolved during training. Every run has one
**tag**, used as the filename stem in `configs/plans/`, `configs/datasets/`,
`configs/splits/` and `results/`:

| Tag | nnU-Net dataset | Notebook | Trainer |
|---|---|---|---|
| `teacher` | `Dataset999_BraTS-GoAT-modified-sub` | `01_teacher/Modified_nnUNet_submission.ipynb` | `nnUNetTrainer_250epochs_Modified` |
| `student_1_labeled_only` | `Dataset952_BraTS-GoAT-base-1fold-supervised-less-val` | `02_students/nnUnet_base_1fold_supervised_less_val.ipynb` | `nnUNetTrainer` |
| `student_2_pseudo75` | `Dataset960_BraTS-GoAT-base-1fold-ssl-75-less-val` | `02_students/nnUnet_base_1fold_ssl_75_less_val.ipynb` | `nnUNetTrainer` |
| `student_3_labeled_aug` | `Dataset961_BraTS-GoAT-base-1fold-labeled-deform-aug-agressive` | `02_students/nnUnet_base_1fold_labeled_deform_aug_agressive.ipynb` | `nnUNetTrainer_deformAug_agressive` |
| `student_4_submitted` | `Dataset959_BraTS-GoAT-base-1fold-ssl-deform-aug-agressive` | `02_students/nnUnet_base_1fold_ssl_deform_aug_agressive.ipynb` | `nnUNetTrainer_deformAug_agressive` |
| `retention_25` | `Dataset956_BraTS-GoAT-base-1fold-ssl-25` | `03_retention_sweep/nnUnet_base_1fold_ssl_25.ipynb` | `nnUNetTrainer` |
| `retention_50` | `Dataset955_BraTS-GoAT-base-1fold-ssl-50` | `03_retention_sweep/nnUnet_base_1fold_ssl_50.ipynb` | `nnUNetTrainer` |
| `retention_75` | `Dataset954_BraTS-GoAT-base-1fold-ssl-75` | `03_retention_sweep/nnUnet_base_1fold_ssl_75.ipynb` | `nnUNetTrainer` |
| `retention_100` | `Dataset953_BraTS-GoAT-base-1fold-ssl-100` | `03_retention_sweep/nnUnet_base_1fold_ssl_100.ipynb` | `nnUNetTrainer` |

The `dataset_name` field inside each `configs/plans/<tag>.json` matches the second column,
so any file can be traced back to the run that produced it.

Every run uses the `nnUNetResEncUNetLPlans` (Large Residual Encoder) 3D full-resolution
configuration, with 1.0 mm isotropic spacing and channel order `T1n, T1c, T2w, T2f`.
The teacher uses a fixed 128×128×128 patch with batch 5 and region-based labels; the
eight student runs all use the 160×192×160 patch with batch 3 that the planner resolved
identically for every dataset. The exact values are in `configs/plans/` and
`configs/datasets/`.

Inference always uses `checkpoint_best.pth` (the epoch with the highest validation Dice),
both in the notebooks and in the container.

### Splits

The labeled pool contains **1,351** cases. Validation is always held out from the
labeled pool only: **no pseudo-labeled case is ever used for validation**.

| Runs | Split of the labeled pool | Train | Val | Pseudo-labels added to train |
|---|---|---|---|---|
| Students (Table 3) | 95% / 5% — `KFold(n_splits=20, seed=12345)`, fold 0 | 1,283 | 68 | 0 or 803 |
| Retention sweep (Table 1) | 80% / 20% — `KFold(n_splits=5, seed=12345)`, fold 0 | 1,080 | 271 | 268 / 536 / 803 / 1,071 |
| Teacher | nnU-Net automatic 5-fold | ~1,081 | ~270 | — |

All four students share one identical 68-case validation set, and all four retention
runs share one identical 271-case validation set, so the rows within each table are
directly comparable. The teacher's five folds cover the 1,351 labeled cases exactly once
each as validation; pseudo-labels are then produced by the 5-fold ensemble applied to the
unlabeled pool, which is disjoint from the labeled data.

## Reproducing

The notebooks were run on an **NVIDIA H100 (80 GB)** with **nnU-Net v2.6.4**, Python 3.10
and PyTorch 2.5.1 (see `configs/environment.txt`). They expect
the official BraTS-GoAT data (not redistributed here) and write into isolated nnU-Net
directories, so experiments never overwrite each other.

Typical order:
1. `01_teacher/Modified_nnUNet_submission.ipynb` — train the teacher (5 folds).
2. `01_teacher/Modified_nnUNet_pseudolabel_generator.ipynb` — generate and rank pseudo-labels.
3. `02_students/*.ipynb` — train each student configuration.
4. `03_retention_sweep/*.ipynb` — the retention study.

## Container (challenge submission)

The container reads case folders from the read-only `/input` and writes exactly one
flat `<case>.nii.gz` per case to `/output`, with **no network access at runtime**.
Case-folder prefixes are handled dynamically (nothing is hard-coded to `BraTS-GoAT`).

```bash
# build (place the weights in docker/model/ first — see docker/model/README.md)
cd docker
docker build -t brats-goat-2026:v1 .

# test locally
docker run --rm --network none --gpus all \
  --volume /ABS/PATH/TO/Validation:/input:ro \
  --volume /ABS/PATH/TO/results:/output:rw \
  --memory=48G --shm-size=16G \
  brats-goat-2026:v1
```

See `docker/README.md` for details.

## Results files

Each CSV in `results/` is the per-case output returned by the challenge platform for one
submission on the official validation set. Columns are `global_dsc_*`, `global_nsd_*` and
`global_hd95_*` for each region (`wt`, `tc`, `et`), plus lesion-wise counts.

| Result file | Produced by | Paper reference |
|---|---|---|
| `teacher.csv` | `notebooks/01_teacher/Modified_nnUNet_submission.ipynb` | Table 3 — Teacher (labeled data only) |
| `student_1_labeled_only.csv` | `notebooks/02_students/nnUnet_base_1fold_supervised_less_val.ipynb` | Table 3 — Student, labeled data only |
| `student_2_pseudo75.csv` | `notebooks/02_students/nnUnet_base_1fold_ssl_75_less_val.ipynb` | Table 3 — Student w/ pseudo-labels |
| `student_3_labeled_aug.csv` | `notebooks/02_students/nnUnet_base_1fold_labeled_deform_aug_agressive.ipynb` | Table 3 — Student w/ tumor-aware aug. |
| `student_4_submitted.csv` | `notebooks/02_students/nnUnet_base_1fold_ssl_deform_aug_agressive.ipynb` | Table 3 — **Student (submitted)** |
| `retention_25.csv` | `notebooks/03_retention_sweep/nnUnet_base_1fold_ssl_25.ipynb` | Table 1 — 25% retention |
| `retention_50.csv` | `notebooks/03_retention_sweep/nnUnet_base_1fold_ssl_50.ipynb` | Table 1 — 50% retention |
| `retention_75.csv` | `notebooks/03_retention_sweep/nnUnet_base_1fold_ssl_75.ipynb` | Table 1 — 75% retention (selected) |
| `retention_100.csv` | `notebooks/03_retention_sweep/nnUnet_base_1fold_ssl_100.ipynb` | Table 1 — 100% retention |
| `pseudolabel_confidence.csv` | `notebooks/01_teacher/Modified_nnUNet_pseudolabel_generator.ipynb` | Sec. 3.2 — confidence ranking used to select the top 75% |
