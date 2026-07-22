# BraTS-GoAT 2026 submission -- nnU-Net v2 (ResEnc-L, deformable-aug trainer).
# Runtime contract: zero network, /input read-only, flat predictions to /output.
# Target compute: NVIDIA A10G (24 GB VRAM), CUDA <= 13.0.

# CUDA 12.4 <= 13.0 and works on A10G. Torch comes preinstalled in this base.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# -- Dependencies (installed at build time; NO network at runtime) --------------
# nnunetv2 is pinned to the training version so the checkpoint loads correctly.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# -- Register the custom trainer -------------------------------------------------
# nnUNetv2_predict imports the trainer class stored in the checkpoint by name,
# so the file must live inside the nnunetv2 package (variants dir).
COPY nnunet_trainer/nnUNetTrainer_deformAug_agressive.py /tmp/trainer.py
RUN python -c "import nnunetv2, os, shutil; \
d=os.path.join(os.path.dirname(nnunetv2.__file__),'training','nnUNetTrainer','variants'); \
shutil.copy('/tmp/trainer.py', os.path.join(d,'nnUNetTrainer_deformAug_agressive.py')); \
print('trainer registered at', d)"

# -- nnU-Net env vars (required even for predict via API) -----------------------
ENV nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results \
    MKL_THREADING_LAYER=GNU \
    nnUNet_compile=0
RUN mkdir -p /tmp/nnUNet_raw /tmp/nnUNet_preprocessed /tmp/nnUNet_results

# -- Trained weights of the submitted model -------------------------------------
# The model tree must be present in ./model before building (see model/README.md).
# Expected: model/DatasetXXX_.../<TRAINER>__<PLANS>__3d_fullres/{plans.json,
#           dataset.json, fold_0/checkpoint_best.pth}
COPY model/ /opt/model/

# -- Inference code --------------------------------------------------------------
COPY run_model.py /opt/run_model.py
WORKDIR /opt

ENTRYPOINT ["python", "run_model.py"]
