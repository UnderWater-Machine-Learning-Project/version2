# UIEB NAFNet Fine-Tuning

This project is a small transfer-learning scaffold for underwater image
enhancement using UIEB and pretrained NAFNet weights.

The goal is not to train from scratch. The intended workflow is:

1. Put the official NAFNet code in `external/NAFNet`.
2. Put pretrained NAFNet weights in `weights/`.
3. Put UIEB image pairs in `data/UIEB`.
4. Train only the restoration head/decoder first.
5. Unfreeze the full network.
6. Fine-tune end-to-end with a low learning rate.

## Folder Layout

```text
configs/              Training settings.
data/UIEB/            Dataset folders. Images are not committed.
external/NAFNet/      Official NAFNet code goes here later.
outputs/              Checkpoints and logs created by training.
src/uieb_finetune/    Small learning-focused Python code.
weights/              Pretrained model weights. Not committed.
```

Expected UIEB layout:

```text
data/UIEB/
  train/
    input/
    target/
  val/
    input/
    target/
```

Each image in `input/` should have the same filename as its clean/reference image
in `target/`.

## First Run Later

After adding NAFNet code, weights, and UIEB:

```powershell
python -m src.uieb_finetune.train --config configs/nafnet_uieb.json
```

Evaluate a trained checkpoint:

```powershell
python -m src.uieb_finetune.evaluate --config configs/nafnet_uieb.json --checkpoint outputs/checkpoints/stage2_full_finetune.pth
```

The code is intentionally small so you can read it line by line and rewrite parts
yourself while learning Python and PyTorch.
