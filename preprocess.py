
# =============================================================================
# preprocess.py — Full Preprocessing Pipeline
# =============================================================================
# Stages (in order):
#   1. UDCP Dehazing         — removes haze using R+G dark channel (physics-based)
#   2. Gray World WB         — fixes residual colour cast AFTER dehaze
#   3. CLAHE on L channel    — mild local contrast boost in LAB space
#   4. Unsharp Mask          — light sharpening, run last to avoid noise amplification
#
# Outputs:
#   output/preprocessed/      — final fully preprocessed images (fed to UWCNN)
#   output/stage_comparison/  — 3x2 grid per image showing every stage:
#
#       Row 1: [ Raw ]  [ After UDCP ]  [ After White Balance ]
#       Row 2: [ After CLAHE ]  [ Final (Sharpened) ]  [ Clear Reference ]
# =============================================================================

import cv2
import numpy as np
import os

# =============================================================================
# FOLDER PATHS
# =============================================================================

RAW_DIR   = "data/UIEB/raw-890"        # read from here
CLEAR_DIR = "data/UIEB/reference-890"  # read references
OUT_FINAL = "data/UIEB/preprocessed-890"  # write here
OUT_STAGES  = "output/stage_comparison"

os.makedirs(OUT_FINAL,  exist_ok=True)
os.makedirs(OUT_STAGES, exist_ok=True)


# =============================================================================
# STAGE 1 — UDCP (Underwater Dark Channel Prior)
# =============================================================================

def udcp_dehaze(img_bgr: np.ndarray,
                omega: float = 0.55,
                patch_size: int = 7,
                t_min: float = 0.35) -> np.ndarray:

    img_f  = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    # Dark channel: min of R and G only (BGR: R=index 2, G=index 1)
    rg_min = np.minimum(img_f[:, :, 2], img_f[:, :, 1])
    dark   = cv2.erode(rg_min, kernel)

    # Background light: top 0.1% haziest pixels in original image
    n_px    = max(1, int(dark.size * 0.001))
    indices = np.unravel_index(np.argsort(dark.ravel())[-n_px:], dark.shape)
    A       = np.clip(img_f[indices[0], indices[1], :].mean(axis=0), 0.3, 1.0)

    # Transmission map
    rg_norm = np.minimum(
        img_f[:, :, 2] / max(A[2], 1e-6),
        img_f[:, :, 1] / max(A[1], 1e-6),
    )
    t = 1.0 - omega * cv2.erode(rg_norm, kernel)
    t = cv2.GaussianBlur(t, (0, 0), 3)   # smooth to remove tile artifacts
    t = np.clip(t, t_min, 1.0)

    # Recover scene: J = (I - A) / t + A
    out = np.empty_like(img_f)
    for c in range(3):
        out[:, :, c] = (img_f[:, :, c] - A[c]) / t + A[c]

    return np.clip(out * 255, 0, 255).astype(np.uint8)


# =============================================================================
# STAGE 2 — GRAY WORLD WHITE BALANCE
# =============================================================================

def white_balance(img_bgr: np.ndarray) -> np.ndarray:

    img_f   = img_bgr.astype(np.float32)
    avg_b   = float(img_f[:, :, 0].mean())
    avg_g   = float(img_f[:, :, 1].mean())
    avg_r   = float(img_f[:, :, 2].mean())
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Scale each channel — clamped to [0.7, 1.4] to prevent extreme corrections
    scale_b = np.clip(avg_all / max(avg_b, 1e-6), 0.7, 1.4)
    scale_g = np.clip(avg_all / max(avg_g, 1e-6), 0.7, 1.4)
    scale_r = np.clip(avg_all / max(avg_r, 1e-6), 0.7, 1.4)

    img_f[:, :, 0] *= scale_b
    img_f[:, :, 1] *= scale_g
    img_f[:, :, 2] *= scale_r

    return np.clip(img_f, 0, 255).astype(np.uint8)


# =============================================================================
# STAGE 3 — CLAHE
# =============================================================================

def clahe_enhance(img_bgr: np.ndarray,
                  clip_limit: float = 1.2,
                  grid: tuple = (8, 8)) -> np.ndarray:

    lab              = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe            = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l_ch             = clahe.apply(l_ch)

    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# =============================================================================
# STAGE 4 — UNSHARP MASK
# =============================================================================

def unsharp_mask(img_bgr: np.ndarray,
                 sigma: float = 0.8,
                 strength: float = 0.20) -> np.ndarray:

    img_f = img_bgr.astype(np.float32)
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    sharp = img_f + strength * (img_f - blur)

    return np.clip(sharp, 0, 255).astype(np.uint8)


# =============================================================================
# 3x2 GRID COMPARISON
# =============================================================================
# Builds a 3-column x 2-row grid:
#
#   Row 1: [ Raw ]   [ After UDCP ]   [ After White Balance ]
#   Row 2: [ After CLAHE ]   [ Final (Sharpened) ]   [ Clear Reference ]
#
# Each cell has a coloured label bar on top so you know which stage is which.

LABELS = [
    ("1. Raw",              (60,  60,  60)),   # dark grey
    ("2. After UDCP",       (150, 80,  10)),   # brown
    ("3. After White Bal.", (10,  100, 150)),   # teal
    ("4. After CLAHE",      (10,  120, 10)),    # green
    ("5. Final (Sharp)",    (10,  10,  160)),   # blue
    ("6. Clear Reference",  (140, 10,  10)),    # red
]

# How wide each label bar is (pixels, height)
LABEL_H    = 30
PANEL_H    = 280   # each image panel is resized to this height
SEP_W      = 5     # pixel gap between panels
SEP_H      = 5     # pixel gap between rows
BG_COLOUR  = (30, 30, 30)   # dark background colour (BGR)


def _make_panel(img_bgr, label_text, label_colour, panel_h):
    """Resize one image to panel_h height and add a coloured label bar on top."""
    h, w     = img_bgr.shape[:2]
    new_w    = max(1, int(w * panel_h / h))      # preserve aspect ratio
    resized  = cv2.resize(img_bgr, (new_w, panel_h))

    # Coloured label bar
    bar = np.full((LABEL_H, new_w, 3), label_colour, dtype=np.uint8)
    cv2.putText(
        bar, label_text,
        (6, 21),                       # x, y position of text
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,                          # font scale
        (255, 255, 255),               # white text
        1, cv2.LINE_AA
    )

    # Stack label bar above the image
    return np.vstack([bar, resized])   # shape: (LABEL_H + panel_h, new_w, 3)


def make_grid_comparison(images_bgr):
    """
    images_bgr: list of 6 BGR images in order:
        [raw, udcp, wb, clahe, final, clear_ref]

    Returns a single image in a 3x2 grid layout.
    """
    assert len(images_bgr) == 6, "Need exactly 6 images"

    panels = []
    for img, (label, colour) in zip(images_bgr, LABELS):
        panels.append(_make_panel(img, label, colour, PANEL_H))

    # Each panel may have slightly different widths due to aspect ratios.
    # Normalise all panels in each row to the same height (already PANEL_H + LABEL_H).
    # Split into two rows of 3
    row1_panels = panels[0:3]
    row2_panels = panels[3:6]

    # Make a thin separator column (vertical gap between panels)
    def hstack_with_sep(panel_list):
        row    = []
        h_max  = max(p.shape[0] for p in panel_list)
        for idx, p in enumerate(panel_list):
            # Pad height if panels differ slightly
            if p.shape[0] < h_max:
                pad = np.full((h_max - p.shape[0], p.shape[1], 3),
                              BG_COLOUR, dtype=np.uint8)
                p   = np.vstack([p, pad])
            row.append(p)
            if idx < len(panel_list) - 1:
                # Add vertical separator
                sep = np.full((h_max, SEP_W, 3), BG_COLOUR, dtype=np.uint8)
                row.append(sep)
        return np.hstack(row)

    row1 = hstack_with_sep(row1_panels)
    row2 = hstack_with_sep(row2_panels)

    # Match widths of the two rows (pad narrower row on the right)
    w1, w2 = row1.shape[1], row2.shape[1]
    if w1 > w2:
        pad = np.full((row2.shape[0], w1 - w2, 3), BG_COLOUR, dtype=np.uint8)
        row2 = np.hstack([row2, pad])
    elif w2 > w1:
        pad = np.full((row1.shape[0], w2 - w1, 3), BG_COLOUR, dtype=np.uint8)
        row1 = np.hstack([row1, pad])

    # Horizontal separator between rows
    h_sep = np.full((SEP_H, row1.shape[1], 3), BG_COLOUR, dtype=np.uint8)

    return np.vstack([row1, h_sep, row2])


# =============================================================================
# PROCESS ALL IMAGES
# =============================================================================

image_files = [
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} images in {RAW_DIR}")
print("Pipeline: UDCP -> White Balance -> CLAHE -> Unsharp Mask")
print("Grid layout: 3 columns x 2 rows per comparison image")

for i, filename in enumerate(image_files):

    in_path = os.path.join(RAW_DIR, filename)
    raw_bgr = cv2.imread(in_path)

    if raw_bgr is None:
        print(f"  [skip] Cannot read: {filename}")
        continue

    # Run each stage individually to capture intermediate outputs
    after_udcp  = udcp_dehaze(raw_bgr,    omega=0.65, patch_size=7, t_min=0.35)
    after_wb    = white_balance(after_udcp)
    after_clahe = clahe_enhance(after_wb,  clip_limit=1.2)
    after_sharp = unsharp_mask(after_clahe, sigma=0.8, strength=0.20)

    # Load matching clear reference image (same filename, different folder)
    stem     = os.path.splitext(filename)[0]
    clear_img = None
    for ext in [".jpg", ".jpeg", ".png"]:
        cp = os.path.join(CLEAR_DIR, stem + ext)
        if os.path.exists(cp):
            clear_img = cv2.imread(cp)
            break

    # If no clear reference found, use a grey placeholder with text
    if clear_img is None:
        clear_img = np.full(raw_bgr.shape, 80, dtype=np.uint8)
        cv2.putText(clear_img, "No reference found",
                    (10, raw_bgr.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # Save final preprocessed image
    cv2.imwrite(os.path.join(OUT_FINAL, filename), after_sharp)

    # Build 3x2 grid comparison and save
    grid = make_grid_comparison([
        raw_bgr,      # panel 1 (top-left)
        after_udcp,   # panel 2 (top-mid)
        after_wb,     # panel 3 (top-right)
        after_clahe,  # panel 4 (bottom-left)
        after_sharp,  # panel 5 (bottom-mid)
        clear_img,    # panel 6 (bottom-right) — ground truth
    ])
    cv2.imwrite(os.path.join(OUT_STAGES, filename), grid)

    if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
        print(f"  Processed {i+1}/{len(image_files)}: {filename}")

print(f"All done!")
print(f"Preprocessed images  → {OUT_FINAL}/")
print(f"Stage comparison grids → {OUT_STAGES}/")
print()
print("Grid layout per image:")
print("  +------------------+------------------+------------------+")
print("  | 1. Raw           | 2. After UDCP    | 3. After WB      |")
print("  +------------------+------------------+------------------+")
print("  | 4. After CLAHE   | 5. Final (Sharp) | 6. Clear Ref     |")
print("  +------------------+------------------+------------------+")
