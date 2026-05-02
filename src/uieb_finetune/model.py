import sys
from pathlib import Path

import torch


def build_nafnet(nafnet_code_dir):
    """Imports NAFNet from the official code that will be added later."""

    code_dir = Path(nafnet_code_dir).resolve()
    if not code_dir.exists():
        raise FileNotFoundError(f"NAFNet code folder not found: {code_dir}")

    sys.path.insert(0, str(code_dir))

    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError as error:
        raise ImportError(
            "Could not import NAFNet. After adding the official repository, "
            "adjust this import in src/uieb_finetune/model.py if needed."
        ) from error

    return NAFNet(img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])


def load_pretrained_weights(model, weights_path, device):
    """Loads pretrained weights while allowing small key mismatches."""

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys while loading weights: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys while loading weights: {len(unexpected_keys)}")

    return model


def freeze_backbone_for_warmup(model):
    """Freezes most layers so early training changes only output-side layers."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_name_hints = ("ending", "tail", "decoder", "upsample")
    for name, parameter in model.named_parameters():
        if any(hint in name.lower() for hint in trainable_name_hints):
            parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
