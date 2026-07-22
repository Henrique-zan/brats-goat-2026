# Trained weights (not tracked in git)

The trained weights of the submitted model are **not** included in this repository
(the checkpoint is ~780 MB). Place the nnU-Net result tree here before building
the container.

Expected layout:

```
docker/model/
└── Dataset959_BraTS-GoAT-base-1fold-ssl-deform-aug-agressive/
    └── nnUNetTrainer_deformAug_agressive__nnUNetResEncUNetLPlans__3d_fullres/
        ├── plans.json
        ├── dataset.json
        └── fold_0/
            └── checkpoint_best.pth
```

Only these three files are required for inference (`checkpoint_final.pth`,
training logs and the `validation/` folder are not needed and only enlarge the image).

`run_model.py` auto-discovers the model folder (it looks for `plans.json` +
`dataset.json` + `fold_0/`), so the exact dataset name does not need to be hard-coded.

The weights are produced by
`notebooks/02_students/nnUnet_base_1fold_ssl_deform_aug_agressive.ipynb`
(the submitted model: top-75% pseudo-labels + tumor-aware augmentation).
