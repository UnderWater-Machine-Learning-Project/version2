import os
import glob
import torch
import argparse
import lpips
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.UIEB import UIEBDataset
from model.UWCNN import UWCNN
from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim, normalize_img


parser = argparse.ArgumentParser(description="Testing UIEB dataset")
parser.add_argument("--model_name", type=str, default="UWCNN")
parser.add_argument("--crop_size",  type=int, default=256)
parser.add_argument("--input_norm", action="store_true")
parser.add_argument("--ckpt_path",  type=str, default=None)
hparams = parser.parse_args()


# ── resolve checkpoint ────────────────────────────────────────────────────────
def find_best_ckpt(ckpt_dir="./checkpoints/UIEB/"):
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
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. Pass --ckpt_path explicitly.")


model_path = find_best_ckpt()
print(f"  Using checkpoint: {model_path}")


# ── output folders ────────────────────────────────────────────────────────────
test_path = f"./data/UIEB/All_Results/{hparams.model_name}/T90/"
pred_path = f"./data/UIEB/All_Results/{hparams.model_name}/C60/"
comp_path = f"./data/UIEB/All_Results/{hparams.model_name}/comparisons/"
os.makedirs(test_path, exist_ok=True)
os.makedirs(pred_path, exist_ok=True)
os.makedirs(comp_path, exist_ok=True)


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
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return (t.unsqueeze(0).cuda() * 2.0) - 1.0


def make_comparison_grid(enhanced_pil, ref_pil, label=""):
    """Stitch enhanced | reference side by side with labels."""
    w, h    = enhanced_pil.size
    ref_pil = ref_pil.resize((w, h))
    pad     = 30
    gap     = 6
    grid    = Image.new("RGB", (w * 2 + gap, h + pad), (20, 20, 20))
    grid.paste(enhanced_pil, (0, pad))
    grid.paste(ref_pil,      (w + gap, pad))
    draw = ImageDraw.Draw(grid)
    draw.text((w // 2 - 35,      4), "Enhanced",  fill=(255, 255, 255))
    draw.text((w + gap + w // 2 - 35, 4), "Reference", fill=(255, 255, 255))
    if label:
        draw.text((4, 4), label, fill=(200, 200, 200))
    return grid


# ── inference: test set (T90) ─────────────────────────────────────────────────
print(f"\nGenerating enhanced images for test set ({len(test_loader)} images) -> {test_path}")
psnr_list, ssim_list, lpips_list = [], [], []

for x, y, filename in tqdm(test_loader, total=len(test_loader)):
    with torch.no_grad():
        x      = x.cuda()
        y_hat  = model(x)
        pred_t = normalize_img(y_hat)

        # ── metrics on raw model output ───────────────────────────────────
        gt_img   = y[0].permute(1, 2, 0).detach().cpu().numpy()
        pred_img = pred_t[0].permute(1, 2, 0).detach().cpu().numpy()

        psnr_val = calc_psnr(pred_img, gt_img, is_for_torch=False)
        psnr_list.append(psnr_val)
        ssim_list.append(calc_ssim(pred_img, gt_img, is_for_torch=False))

        pred_uint8 = (pred_img * 255).clip(0, 255).astype(np.uint8)
        gt_uint8   = (gt_img   * 255).clip(0, 255).astype(np.uint8)
        lpips_list.append(lpips_fn(to_lpips_tensor(pred_uint8), to_lpips_tensor(gt_uint8)).item())

        # ── post-process for visual output ────────────────────────────────
        pred_pil = TF.to_pil_image(pred_t[0].cpu())
        pred_pil = TF.adjust_contrast(pred_pil,   contrast_factor=1.3)
        pred_pil = TF.adjust_saturation(pred_pil, saturation_factor=1.1)
        pred_pil = TF.adjust_sharpness(pred_pil,  sharpness_factor=1.1)

        # save enhanced image
        save_image(TF.to_tensor(pred_pil), os.path.join(test_path, filename[0]), normalize=False)

        # save side-by-side comparison grid
        ref_pil  = TF.to_pil_image(y[0].cpu())
        label    = f"PSNR: {psnr_val:.2f} dB"
        grid_img = make_comparison_grid(pred_pil, ref_pil, label=label)
        grid_img.save(os.path.join(comp_path, filename[0]))

print(f"\n  {'Metric':<10} {'Score':>10}")
print(f"  {'PSNR':<10} {sum(psnr_list)  / len(psnr_list):>10.4f} dB")
print(f"  {'SSIM':<10} {sum(ssim_list)  / len(ssim_list):>10.4f}")
print(f"  {'LPIPS':<10} {sum(lpips_list) / len(lpips_list):>10.4f}  (lower=better)")


# ── test results plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Test Results — {hparams.model_name} on UIEB T90", fontsize=13, fontweight="bold")
img_ids = range(1, len(psnr_list) + 1)

axes[0].plot(img_ids, psnr_list, "g-", linewidth=1, alpha=0.7)
axes[0].axhline(sum(psnr_list)/len(psnr_list), color='darkgreen', linestyle='--', linewidth=1.5, label=f"Avg: {sum(psnr_list)/len(psnr_list):.4f} dB")
axes[0].set_title("PSNR (dB)"); axes[0].set_xlabel("Image"); axes[0].set_ylabel("PSNR")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(img_ids, ssim_list, "b-", linewidth=1, alpha=0.7)
axes[1].axhline(sum(ssim_list)/len(ssim_list), color='darkblue', linestyle='--', linewidth=1.5, label=f"Avg: {sum(ssim_list)/len(ssim_list):.4f}")
axes[1].set_title("SSIM"); axes[1].set_xlabel("Image"); axes[1].set_ylabel("SSIM")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(img_ids, lpips_list, "r-", linewidth=1, alpha=0.7)
axes[2].axhline(sum(lpips_list)/len(lpips_list), color='darkred', linestyle='--', linewidth=1.5, label=f"Avg: {sum(lpips_list)/len(lpips_list):.4f}")
axes[2].set_title("LPIPS (lower=better)"); axes[2].set_xlabel("Image"); axes[2].set_ylabel("LPIPS")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("test_results.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  [Plot] Saved -> test_results.png")
print(f"  [Grid] Comparisons saved -> {comp_path}")


# ── inference: challenging set (C60) ─────────────────────────────────────────
print(f"\nGenerating enhanced images for challenging set ({len(pred_loader)} images) -> {pred_path}")

for x, y, filename in tqdm(pred_loader, total=len(pred_loader)):
    with torch.no_grad():
        x      = x.cuda()
        y_hat  = model(x)
        pred_t = normalize_img(y_hat)

        pred_pil = TF.to_pil_image(pred_t[0].cpu())
        pred_pil = TF.adjust_contrast(pred_pil,   contrast_factor=1.4)
        pred_pil = TF.adjust_saturation(pred_pil, saturation_factor=1.5)
        pred_pil = TF.adjust_sharpness(pred_pil,  sharpness_factor=1.2)
        save_image(TF.to_tensor(pred_pil), os.path.join(pred_path, filename[0]), normalize=False)

print("\nAll done!")
print(f"  T90 results   -> {test_path}")
print(f"  C60 results   -> {pred_path}")
print(f"  Comparisons   -> {comp_path}")
print(f"  Metrics plot  -> test_results.png")
