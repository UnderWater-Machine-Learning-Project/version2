import math

import torch


def psnr(prediction, target, max_value=1.0):
    """Peak signal-to-noise ratio for images scaled between 0 and 1."""

    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return float("inf")

    return 20 * math.log10(max_value / math.sqrt(mse))
