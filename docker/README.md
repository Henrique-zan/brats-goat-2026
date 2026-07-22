# BraTS-GoAT 2026 -- Docker submission (nnU-Net v2 student model)

Containerizes the submitted model (`nnUnet_base_1fold_ssl_deform_aug_agressive`:
ResEnc-L nnU-Net, 3d_fullres, fold 0, deformable-aug trainer, best checkpoint).

Runtime contract enforced by the challenge:
- **zero network** at runtime (all deps installed at build time);
- `/input` is **read-only** (we only symlink from it, never write);
- predictions written to `/output` as a **flat** structure: one `<input-folder-name>.nii.gz`
  per input case folder, **no sub-folders**. The case prefix is taken from the input
  folder names, so nothing is hard-coded to `BraTS-GoAT`.

## Files
| File | Role |
|---|---|
| `Dockerfile` | base = pytorch/cuda; installs `nnunetv2`; registers the custom trainer; copies weights + code |
| `run_model.py` | iterates `/input`, builds the 4 channels (t1n,t1c,t2w,t2f), runs `nnUNetPredictor`, writes flat `/output` |
| `requirements.txt` | `nnunetv2` pin — `2.6.4`, the version used for training |
| `nnunet_trainer/nnUNetTrainer_deformAug_agressive.py` | custom trainer (predict imports it by name from the checkpoint) |
| `model/` | put the trained weights here (see `model/README.md`) |
| `compose.yaml`, `.env.example` | convenience build/test wrappers |

## Before building
1. **Weights**: place the trained model tree in `model/` (see `model/README.md`).
2. **.env**: `cp .env.example .env` and set `IMAGE`, `INPUT_DIR`, `OUTPUT_DIR`.

`requirements.txt` pins `nnunetv2==2.6.4`, the version used for training
(see `../configs/environment.txt`), matching the torch 2.5.1 base image.

## Build
```sh
docker compose build
# or, without compose:
docker build -t brats-goat-2026:v1 .
```

## Test locally (mimics the evaluation environment)
Use the GoAT validation set as `/input`. Exactly the challenge's own command:
```sh
docker run --rm \
  --network none \
  --gpus all \
  --volume /ABS/PATH/TO/GoAT_validation:/input:ro \
  --volume /ABS/PATH/TO/results:/output:rw \
  --memory=48G --shm-size=16G \
  brats-goat-2026:v1
```
Then verify the output is **flat** and complete:
```sh
ls /ABS/PATH/TO/results        # should be BraTS-GoAT-XXXXX.nii.gz files, no sub-folders
```
Sanity checks after a run:
- one `.nii.gz` per input case folder;
- no sub-directories under `/output`;
- `/input` unchanged (read-only).

## Push to the Synapse registry
```sh
docker login docker.synapse.org --username <user>   # password = Synapse PAT (Modify)
docker compose push        # or: docker push docker.synapse.org/<PROJECT_ID>/<name>:<tag>
```

## Notes
- Confirm the GPU is visible inside the container:
  `docker run --rm --gpus all brats-goat-2026:v1 python -c "import torch;print(torch.cuda.is_available())"`
- If the GPU runs out of VRAM, run with `NNUNET_ALL_ON_DEVICE=0` (no code change needed).
  A full 451-case run completed on a 12 GB card with everything on device, so the
  24 GB A10G has ample headroom.
- Output naming is `<input-folder-name>.nii.gz`, derived from the input folders.
