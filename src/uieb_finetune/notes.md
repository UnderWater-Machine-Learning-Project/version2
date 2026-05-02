# How This Transfer-Learning Code Works

## Dataset

`PairedUIEBDataset` reads two images:

- `underwater`: the degraded UIEB input image
- `target`: the clean/reference image we want the model to produce

Both are resized, converted to PyTorch tensors, and returned as one training
sample.

## Model

`build_nafnet()` will import the official NAFNet architecture after we add the
official code into `external/NAFNet`.

NAFNet is an image restoration network. For UIEB, the input and output are both
RGB images, so the usual shape is:

```text
input:  batch x 3 x height x width
output: batch x 3 x height x width
```

That means we usually do not need to replace the final head unless the imported
pretrained model has a different output setup.

## Pretrained Weights

`load_pretrained_weights()` loads learned parameters from a `.pth` file.

This is the transfer-learning part: the network starts from useful image
restoration knowledge instead of random weights.

## Freezing

`freeze_backbone_for_warmup()` sets most parameters to:

```python
requires_grad = False
```

Frozen parameters still run during forward pass, but they are not updated during
backpropagation.

This lets the later/output-side layers adjust to UIEB first.

## Fine-Tuning

After warmup, `unfreeze_all()` makes every parameter trainable again.

Then the whole model is trained with a lower learning rate so the pretrained
knowledge is adjusted gently instead of overwritten.

## Loss

The training script uses `L1Loss`.

For image restoration, this means:

```text
average absolute difference between predicted pixels and target pixels
```

Lower loss means the enhanced image is closer to the reference image.

## Evaluation

`evaluate.py` loads a trained checkpoint, runs the model on validation images,
saves enhanced outputs, and reports PSNR.

PSNR is not perfect, but it is a standard first metric for image restoration.
Higher PSNR usually means the prediction is closer to the reference image.
