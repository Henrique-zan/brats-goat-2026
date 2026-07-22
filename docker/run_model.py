#!/usr/bin/env python
"""BraTS-GoAT 2026 inference entrypoint (submitted nnU-Net v2 student model).

Reads each case folder from /input (read-only), runs the trained ResEnc-L
nnU-Net (deformable-aug trainer, fold 0, best checkpoint) and writes exactly
ONE flat prediction per case to /output as <case>.nii.gz -- no sub-folders,
nothing written back to /input.
"""
import os
import shutil
from pathlib import Path

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/opt/model"))
CHECKPOINT = os.environ.get("NNUNET_CHECKPOINT", "checkpoint_best.pth")
FOLD = int(os.environ.get("NNUNET_FOLD", "0"))
# Default True (fits the 24 GB A10G). Set NNUNET_ALL_ON_DEVICE=0 for local testing
# on smaller GPUs (does the sliding-window aggregation on CPU -> less VRAM, a bit slower).
PERFORM_ON_DEVICE = os.environ.get("NNUNET_ALL_ON_DEVICE", "1") != "0"

# Channels in the SAME order used at training (dataset.json: 0=T1n,1=T1c,2=T2w,3=T2f).
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def find_model_folder(root: Path) -> Path:
    """Locate the TRAINER__PLANS__CONFIG folder (plans.json + dataset.json + fold_X/)."""
    for plans in sorted(root.rglob("plans.json")):
        folder = plans.parent
        if (folder / "dataset.json").exists() and (folder / f"fold_{FOLD}").is_dir():
            return folder
    raise FileNotFoundError(f"No trained model (plans.json + fold_{FOLD}/) found under {root}")


def find_modality(case_dir: Path, key: str) -> Path:
    for cand in (case_dir / f"{key}.nii.gz", case_dir / f"{key}.nii"):
        if cand.exists():
            return cand
    cands = sorted(case_dir.glob(f"*{key}*.nii*"))
    if key == "t1n":  # do not match t1c when looking for t1n
        cands = [c for c in cands if "t1c" not in c.name.lower()]
    if not cands:
        raise FileNotFoundError(f"Modality '{key}' not found in {case_dir}")
    return cands[0]


def stage_inputs(input_dir: Path, staging: Path) -> list:
    """Create {case}_{ch:04d}.nii.gz links/copies for nnU-Net (never writes to /input)."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    cases = sorted(d for d in input_dir.iterdir() if d.is_dir())
    for case_dir in cases:
        cid = case_dir.name
        for ch, mod in enumerate(MODALITIES):
            src = find_modality(case_dir, mod)
            dst = staging / f"{cid}_{ch:04d}.nii.gz"
            try:
                os.symlink(src, dst)          # symlink -> does not touch /input
            except OSError:
                shutil.copy2(src, dst)        # fallback if symlinks are unavailable
    return [d.name for d in cases]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path("/tmp/nnunet_input")
    case_ids = stage_inputs(INPUT_DIR, staging)
    print(f"[goat] {len(case_ids)} case(s) found in {INPUT_DIR}", flush=True)

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder = find_model_folder(MODEL_ROOT)
    print(f"[goat] model: {model_folder} | fold {FOLD} | {CHECKPOINT}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=PERFORM_ON_DEVICE,   # NNUNET_ALL_ON_DEVICE=0 -> CPU aggregation
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=(FOLD,), checkpoint_name=CHECKPOINT,
    )
    predictor.predict_from_files(
        str(staging), str(OUTPUT_DIR),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0,
    )

    # nnU-Net also drops plans.json / dataset.json / predict_from_raw_data_args.json
    # into the output folder. The challenge requires a FLAT /output with only the
    # per-case predictions, so keep just the expected {case}.nii.gz files.
    expected = {f"{cid}.nii.gz" for cid in case_ids}
    for item in OUTPUT_DIR.iterdir():
        if item.name not in expected:
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    n_out = len(list(OUTPUT_DIR.glob("*.nii.gz")))
    print(f"[goat] {n_out} flat prediction(s) written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
