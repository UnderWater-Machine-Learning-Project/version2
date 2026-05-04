import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

RAW_DIR   = "data/UIEB/raw-890"
OUT_FINAL = "data/UIEB/preprocessed-890"
os.makedirs(OUT_FINAL, exist_ok=True)


def udcp_dehaze(img_bgr, omega=0.55, patch_size=7, t_min=0.35):
    img_f  = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    rg_min = np.minimum(img_f[:, :, 2], img_f[:, :, 1])
    dark   = cv2.erode(rg_min, kernel)
    n_px    = max(1, int(dark.size * 0.001))
    indices = np.unravel_index(np.argsort(dark.ravel())[-n_px:], dark.shape)
    A       = np.clip(img_f[indices[0], indices[1], :].mean(axis=0), 0.3, 1.0)
    rg_norm = np.minimum(
        img_f[:, :, 2] / max(A[2], 1e-6),
        img_f[:, :, 1] / max(A[1], 1e-6),
    )
    t = 1.0 - omega * cv2.erode(rg_norm, kernel)
    t = cv2.GaussianBlur(t, (0, 0), 3)
    t = np.clip(t, t_min, 1.0)
    out = np.empty_like(img_f)
    for c in range(3):
        out[:, :, c] = (img_f[:, :, c] - A[c]) / t + A[c]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def white_balance(img_bgr):
    img_f   = img_bgr.astype(np.float32)
    avg_b   = float(img_f[:, :, 0].mean())
    avg_g   = float(img_f[:, :, 1].mean())
    avg_r   = float(img_f[:, :, 2].mean())
    avg_all = (avg_b + avg_g + avg_r) / 3.0
    scale_b = np.clip(avg_all / max(avg_b, 1e-6), 0.7, 1.4)
    scale_g = np.clip(avg_all / max(avg_g, 1e-6), 0.7, 1.4)
    scale_r = np.clip(avg_all / max(avg_r, 1e-6), 0.7, 1.4)
    img_f[:, :, 0] *= scale_b
    img_f[:, :, 1] *= scale_g
    img_f[:, :, 2] *= scale_r
    return np.clip(img_f, 0, 255).astype(np.uint8)


def clahe_enhance(img_bgr, clip_limit=1.2, grid=(8, 8)):
    lab              = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe            = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l_ch             = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def unsharp_mask(img_bgr, sigma=0.8, strength=0.20):
    img_f = img_bgr.astype(np.float32)
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    sharp = img_f + strength * (img_f - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess(img_bgr):
    out = udcp_dehaze(img_bgr)
    out = white_balance(out)
    out = clahe_enhance(out)
    out = unsharp_mask(out)
    return out


def process_one(args):
    src_path, dst_path = args
    try:
        img = cv2.imread(src_path)
        if img is None:
            return f"SKIP: {os.path.basename(src_path)}"
        cv2.imwrite(dst_path, preprocess(img))
        return f"OK: {os.path.basename(src_path)}"
    except Exception as e:
        return f"ERROR {os.path.basename(src_path)}: {e}"


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")) +
                   glob.glob(os.path.join(RAW_DIR, "*.jpg")) +
                   glob.glob(os.path.join(RAW_DIR, "*.jpeg")))

    tasks  = [(p, os.path.join(OUT_FINAL, os.path.basename(p))) for p in paths]
    ncores = cpu_count()

    print(f"Found {len(tasks)} images | Using {ncores} CPU cores")

    done = 0
    with Pool(processes=ncores) as pool:
        for result in pool.imap_unordered(process_one, tasks):
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {result}")

    print(f"\nDone! Preprocessed images saved to: {OUT_FINAL}/")
