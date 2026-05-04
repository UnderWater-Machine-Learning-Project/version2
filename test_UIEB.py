import os
import glob
import torch
import argparse
import lpips
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.UIEB import UIEBDataset
from model.UWCNN import UWCNN
from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim, normalize_img


parser = argparse.ArgumentParser(description="Testing UIEB dataset")
parser.add_argument("--model_name", type=str, default="UWCNN",
                    help="model name, options: [UIEC2Net, UTrans, NU2Net, UWCNN, FIVE_APLUS]")
parser.add_argument("--crop_size",  type=int, default=256, help="crop size")
parser.add_argument("--input_norm", action="store_true", help="norm input to [-1,1]")
parser.add_argument("--ckpt_path",  type=str, default=None,
                    help="explicit checkpoint path. If not set, auto-picks best PSNR checkpoint.")
hparams = parser.parse_args()


# ── resolve checkpoint ────────────────────────────────────────────────────────
def find_best_ckpt(model_name, ckpt_dir="./checkpoints/UIEB/"):
    if hparams.ckpt_path and os.path.isfile(hparams.ckpt_path):
        return hparams.ckpt_path

    epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "epoch*-psnr*-ssim*.ckpt"))
    if epoch_ckpts:
        def psnr_from_name(p):
            try:
                return float(os.path.basename(p).split("psnr")[1].split("-")[0])
            except Exception:
                return 0.0
        epoch_ckpts.sort(key=psnr_from_name, reverse=True)
        return epoch_ckpts[0]

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. Pass --ckpt_path explicitly."
    )


model_path = find_best_ckpt(hparams.model_name)
print(f"  Using checkpoint: {model_path}")


# ── output folders ────────────────────────────────────────────────────────────
test_path = f"./data/UIEB/All_Results/{hparams.model_name}/T90/"
pred_path = f"./data/UIEB/All_Results/{hparams.model_name}/C60/"
os.makedirs(test_path, exist_ok=True)
os.makedirs(pred_path, exist_ok=True)


# ── datasets ──────────────────────────────────────────────────────────────────
test_set = UIEBDataset(
    "./data/", train_flag=False, pred_flag=False,
    train_size=hparams.crop_size, input_norm=hparams.input_norm
)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

pred_set = UIEBDataset(
    "./data/", train_flag=False, pred_flag=True,
    train_size=hparams.crop_size, input_norm=hparams.input_norm
)
pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False)


# ── model ─────────────────────────────────────────────────────────────────────
model = UWCNN().cuda()
ckpt  = torch.load(model_path, map_location="cuda")
state = ckpt.get("state_dict", ckpt)
new_state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(new_state, strict=False)
print(f"  Missing keys    : {missing}")
print(f"  Unexpected keys : {unexpected}")
model.eval()


# ── LPIPS setup ───────────────────────────────────────────────────────────────
lpips_fn = lpips.LPIPS(net='alex').cuda()


def to_lpips_tensor(img_np):
    """Convert HxWxC uint8 numpy → (1,3,H,W) float tensor in [-1,1]"""
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return (t.unsqueeze(0).cuda() * 2.0) - 1.0


# ── inference: test set (T90) ─────────────────────────────────────────────────
print(f"\nGenerating enhanced images for test set ({len(test_loader)} images) -> {test_path}")
psnr_list, ssim_list, lpips_list = [], [], []

for x, y, filename in tqdm(test_loader, total=len(test_loader)):
    with torch.no_grad():
        x     = x.cuda()
        y_hat = model(x)

        gt_img   = y[0].permute(1, 2, 0).detach().cpu().numpy()
        pred_t   = normalize_img(y_hat)
        pred_img = pred_t[0].permute(1, 2, 0).detach().cpu().numpy()

        psnr_list.append(calc_psnr(pred_img, gt_img, is_for_torch=False))
        ssim_list.append(calc_ssim(pred_img, gt_img, is_for_torch=False))

        # LPIPS — convert numpy uint8 arrays to [-1,1] tensors
        pred_uint8 = (pred_img * 255).clip(0, 255).astype(np.uint8)
        gt_uint8   = (gt_img   * 255).clip(0, 255).astype(np.uint8)
        lpips_list.append(lpips_fn(to_lpips_tensor(pred_uint8), to_lpips_tensor(gt_uint8)).item())

        save_image(pred_t[0], os.path.join(test_path, filename[0]), normalize=False)

print(f"\n  {'Metric':<10} {'Score':>10}")
print(f"  {'PSNR':<10} {sum(psnr_list)  / len(psnr_list):>10.4f} dB")
print(f"  {'SSIM':<10} {sum(ssim_list)  / len(ssim_list):>10.4f}")
print(f"  {'LPIPS':<10} {sum(lpips_list) / len(lpips_list):>10.4f}  (lower=better)")


# ── inference: challenging set (C60) ─────────────────────────────────────────
print(f"\nGenerating enhanced images for challenging set ({len(pred_loader)} images) -> {pred_path}")

for x, y, filename in tqdm(pred_loader, total=len(pred_loader)):
    with torch.no_grad():
        x      = x.cuda()
        y_hat  = model(x)
        pred_t = normalize_img(y_hat)
        save_image(pred_t[0], os.path.join(pred_path, filename[0]), normalize=False)

print("\nAll done!")
print(f"  T90 results -> {test_path}")
print(f"  C60 results -> {pred_path}")
