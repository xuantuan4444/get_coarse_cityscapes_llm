import json
import math
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no X server needed on a remote VM
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights
from tqdm.auto import tqdm


# ============================================================================
# CONFIG -- every path is resolved from PROJECT_ROOT (this script's own folder), never from the
# current working directory and never from an environment variable.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

CITYSCAPES_ROOT = PROJECT_ROOT / "data_cityscapes"
CITYSCAPES_TRAIN_ROOT = CITYSCAPES_ROOT / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train"
CITYSCAPES_GT_ROOT = CITYSCAPES_ROOT / "gtFine_trainvaltest" / "gtFine" / "train"

ADJUST_PROMPT_PATH = PROJECT_ROOT / "adjust_prompt_city_v3.json"
COARSE_CACHE_DIR = PROJECT_ROOT / "coarse_cache_city_llm"
SPLIT_JSON = COARSE_CACHE_DIR / "split.json"

DINOV2_CACHE_DIR = PROJECT_ROOT / "dinov2_cache_city"

WEIGHTS_DIR = PROJECT_ROOT / "weights_unetasppdino_city_llm_v1"
BEST_CKPT = WEIGHTS_DIR / "unet_aspp_dino_city_llm_v1_best.pth"
LAST_CKPT = WEIGHTS_DIR / "unet_aspp_dino_city_llm_v1_last.pth"
HISTORY_JSON = WEIGHTS_DIR / "unet_aspp_dino_city_llm_v1_history.json"
PLOT_PATH = PROJECT_ROOT / "training_curves_unetasppdinov2_city_llm.png"

CITYSCAPES_CLASSES_ALL = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
GT_CLASS_TO_TRAINID = {name: idx for idx, name in enumerate(CITYSCAPES_CLASSES_ALL)}  # 0..18

# ─── Target classes: single source of truth is adjust_prompt.json ─────
with open(ADJUST_PROMPT_PATH, encoding="utf-8") as f:
    ADJUST_PROMPT = json.load(f)

TARGET_CLASSES = list(ADJUST_PROMPT.keys())
TARGET_TO_TRAINID = {name: GT_CLASS_TO_TRAINID[name] for name in TARGET_CLASSES}
TARGET_TO_LOCAL = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
LOCAL_TO_TARGET = {v: k for k, v in TARGET_TO_LOCAL.items()}
NUM_TARGETS = len(TARGET_CLASSES)      # 14
NUM_OUT_CLASSES = NUM_TARGETS + 1      # 15  (ch0 = other/bg)

# ─── DINOv2 config
DINOV2_MODEL_NAME = "dinov2_vits14"
DINOV2_EMBED_DIM = 384
DINOV2_PATCH_SIZE = 14
DINOV2_INPUT_SIZE = 518    # resize for DINOv2 (square)
DINOV2_GRID_SIZE = DINOV2_INPUT_SIZE // DINOV2_PATCH_SIZE  # 37
DINOV2_COMPRESS = 32

# ─── Input channel design (IMAGE_H x IMAGE_W) ───────────────────────────────
# RGB(3) + NUM_TARGETS coarse = 17ch (dataset) -> model adds 32ch DINOv2 upsample -> 49ch total.
IN_CHANNELS_BASE = 3 + NUM_TARGETS                       # 17
IN_CHANNELS_TOTAL = IN_CHANNELS_BASE + DINOV2_COMPRESS   # 49

# ─── original scale Cityscapes (2048x1024 = 2:1) ──────────────
IMAGE_H = 512
IMAGE_W = 1024

# ─── Hyperparameters ───────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 2      # 512x1024 x 49ch -> reduce batch
EPOCHS = 30
WARMUP_EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4     # Ubuntu VM, not Windows -> no need for the os.name=='nt' guard
PATIENCE = 8

# Per-class Tversky (alpha=penalize FP, beta=penalize FN). Class thin/easily confused
# (rider, pole, train, bicycle, motorcycle, traffic sign/light) --> recall
# ; large area (sidewalk/vegetation/terrain/wall/person) maintain balance.
TVERSKY_PARAMS = {
    "sidewalk": (0.50, 0.50),
    "vegetation": (0.50, 0.50),
    "terrain": (0.50, 0.50),
    "wall": (0.50, 0.50),
    "person": (0.50, 0.50),
    "fence": (0.45, 0.55),
    "bus": (0.45, 0.55),
    "pole": (0.40, 0.60),
    "train": (0.40, 0.60),
    "bicycle": (0.35, 0.65),
    "motorcycle": (0.35, 0.65),
    "traffic sign": (0.35, 0.65),
    "traffic light": (0.35, 0.65),
    "rider": (0.35, 0.65),
}

# Boundary loss weight -- high for object thin/complex (pole, traffic sign/light, rider,
# bicycle, motorcycle), low for large area (vegetation, terrain) because boundaries are less
# important.
BOUNDARY_LAMBDA = {
    "vegetation": 0.5,
    "terrain": 0.5,
    "sidewalk": 1.0,
    "wall": 1.0,
    "person": 1.0,
    "bus": 1.0,
    "train": 1.0,
    "fence": 1.5,
    "rider": 2.0,
    "bicycle": 2.0,
    "motorcycle": 2.0,
    "pole": 2.5,
    "traffic sign": 2.5,
    "traffic light": 2.5,
}

RIDER_CE_BOOST = 1.5
FOCAL_GAMMA = 1.5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Raw Cityscapes labelId -> trainId table (cityscapesscripts/helpers/labels.py)
ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18, -1: 255,
}
_ID2TRAIN_LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in ID_TO_TRAINID.items():
    if _k >= 0:
        _ID2TRAIN_LUT[_k] = _v

DEVICE = None  # set in setup_torch()

# Set inside compute_ce_weights(); referenced as free variables by combined_loss()/per_class_loss(),
# exactly like the notebook's cell-global scope.
CE_WEIGHTS = None
BOUNDARY_LAMBDA_TENSOR = None
TVERSKY_TENSOR = None

# Set inside build_dinov2_cache(); dinov2_forward() (module-level) reads it as a free variable.
dinov2_model = None


# ============================================================================
# Environment setup
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch() -> torch.device:
    global DEVICE
    print("Albumentations:", A.__version__)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device  :", DEVICE)
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))

    print(f"Target classes (local 1..{NUM_TARGETS}): {TARGET_CLASSES}")
    print("Output classes: other/bg(0) + " + ", ".join(f"{c}({i})" for i, c in enumerate(TARGET_CLASSES, 1)))
    print(f"Image size    : {IMAGE_H} x {IMAGE_W}  (H x W)")
    print(f"DINOv2        : {DINOV2_MODEL_NAME}, embed_dim={DINOV2_EMBED_DIM}, "
          f"input={DINOV2_INPUT_SIZE}x{DINOV2_INPUT_SIZE}, grid={DINOV2_GRID_SIZE}x{DINOV2_GRID_SIZE}")
    print(f"IN_CHANNELS   : {IN_CHANNELS_BASE} (dataset) -> {IN_CHANNELS_TOTAL} (after DINOv2 concat in model)")
    print(f"OUT_CLASSES   : {NUM_OUT_CLASSES}  (other/bg + {NUM_TARGETS} target)")
    print(f"Batch size    : {BATCH_SIZE}   Epochs: {EPOCHS} (warmup={WARMUP_EPOCHS} + cosine)   Patience: {PATIENCE}")
    print(f"Rider CE boost : x{RIDER_CE_BOOST}")
    print(f"Focal gamma    : {FOCAL_GAMMA}")
    return DEVICE


# ============================================================================
# Split loading + coarse-cache / GT validation
# ============================================================================

def _resolve(win_path_str, root):
    p = PureWindowsPath(win_path_str)
    city = p.parts[-2]
    fname = p.parts[-1]
    return root / city / fname


def _coarse_ok(stem):
    p = COARSE_CACHE_DIR / f"{stem}.npz"
    if not p.exists():
        return False
    try:
        with np.load(p) as d:
            return set(TARGET_CLASSES) - set(d.files) == set()
    except Exception:
        return False


def check_ready(img_path):
    stem = img_path.name.replace("_leftImg8bit.png", "")
    if not img_path.exists():
        return False, "missing_image"
    if not _coarse_ok(stem):
        return False, "missing_or_bad_coarse_npz"
    return True, "ok"


def filter_ready(image_paths, tag):
    ok_list, skipped = [], []
    for p in tqdm(image_paths, desc=f"Validate {tag}"):
        ok, reason = check_ready(p)
        (ok_list if ok else skipped).append(p if ok else (p, reason))
    print(f"  {tag}: usable {len(ok_list)}/{len(image_paths)}")
    for item in skipped[:5]:
        print(f"    skipped: {item}")
    return ok_list


def get_gt_mask_path(img_path):
    """.../leftImg8bit/train/<city>/<city>_..._leftImg8bit.png
       -> .../gtFine/train/<city>/<city>_..._gtFine_labelTrainIds.png (or _labelIds.png)"""
    city = img_path.parent.name
    stem = img_path.name.replace("_leftImg8bit.png", "")
    mask_dir = CITYSCAPES_GT_ROOT / city
    train_id_path = mask_dir / f"{stem}_gtFine_labelTrainIds.png"
    label_id_path = mask_dir / f"{stem}_gtFine_labelIds.png"
    if train_id_path.exists():
        return train_id_path, True
    elif label_id_path.exists():
        return label_id_path, False
    return None, None


def load_cityscapes_trainid_mask(mask_path, is_train_id):
    """Returns the raw trainId map (0-18, 255=ignore) without shifting by +1; the training
    label map will assign local IDs (1..NUM_TARGETS) to each target class, while everything
    else (including the 255s remaining after valid filtering) is other/background (0)."""
    raw = np.array(Image.open(mask_path), dtype=np.uint8)
    return raw if is_train_id else _ID2TRAIN_LUT[raw]


def filter_has_gt(image_paths, tag):
    ok, missing = [], []
    for p in image_paths:
        mp, _ = get_gt_mask_path(p)
        (ok if mp is not None else missing).append(p)
    print(f"{tag}: GT found {len(ok)}/{len(image_paths)}  (missing {len(missing)})")
    return ok


def load_and_validate_split():
    """Load internal train/val split (produced by get_coarse_city_llm.py) and filter to images
    that have both a valid coarse-cache entry and a matching GT mask."""
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    internal_train_images = [_resolve(s, CITYSCAPES_TRAIN_ROOT) for s in split["train"]]
    internal_val_images = [_resolve(s, CITYSCAPES_TRAIN_ROOT) for s in split["val"]]

    print(f"Internal train (raw from split.json): {len(internal_train_images)}")
    print(f"Internal val   (raw from split.json): {len(internal_val_images)}")

    internal_train_images = filter_ready(internal_train_images, "internal-train")
    internal_val_images = filter_ready(internal_val_images, "internal-val")

    if not internal_train_images:
        raise RuntimeError("No usable internal-train images. Check CITYSCAPES_TRAIN_ROOT / COARSE_CACHE_DIR paths.")

    internal_train_images = filter_has_gt(internal_train_images, "internal-train")
    internal_val_images = filter_has_gt(internal_val_images, "internal-val")

    print("\u2713 GT mask helpers defined")
    return internal_train_images, internal_val_images


# ============================================================================
# DINOv2 cache (built once, for internal-train + internal-val)
# ============================================================================

def load_dinov2_model():
    """Load frozen DINOv2, trying torch.hub first, then a local-checkout fallback -- identical
    logic to the notebook's fallback chain, for VMs without internet access to GitHub."""
    print("Loading DINOv2...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL_NAME, trust_repo=True)
    except Exception as e:
        print(f"torch.hub.load failed: {e}")
        print("Fallback: trying a local DINOv2 checkout...")
        dinov2_repo_candidates = [
            PROJECT_ROOT / "dinov2-repo",
            Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
        ]
        repo_path = next((p for p in dinov2_repo_candidates if p.exists()), None)
        if repo_path is None:
            raise RuntimeError(
                f"DINOv2 repo not found. Tried: {dinov2_repo_candidates}. "
                "Please enable internet access, or place a local checkout at one of those paths."
            )
        sys.path.insert(0, str(repo_path))
        from dinov2.hub.backbones import dinov2_vits14 as _dinov2_build
        model = _dinov2_build(pretrained=False)
        w_candidates = [
            PROJECT_ROOT / "dinov2_vits14_pretrain.pth",
            repo_path / "dinov2_vits14_pretrain.pth",
        ]
        wpath = next((p for p in w_candidates if p.exists()), None)
        if wpath is None:
            raise RuntimeError(f"DINOv2 weights not found. Tried: {w_candidates}")
        sd = torch.load(wpath, map_location="cpu")
        model.load_state_dict(sd)
        print(f"Loaded DINOv2 weights from {wpath}")

    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DINOv2 {DINOV2_MODEL_NAME} loaded ({n_params:,} params, frozen)")
    return model


@torch.no_grad()
def dinov2_forward(image_pil):
    img_r = image_pil.convert("RGB").resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.BILINEAR)
    x = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(DEVICE)
    out = dinov2_model.forward_features(x)
    tokens = out["x_norm_patchtokens"]  # [1, 1369, 384]
    B, N, D = tokens.shape
    assert N == DINOV2_GRID_SIZE * DINOV2_GRID_SIZE
    assert D == DINOV2_EMBED_DIM
    feat = tokens.transpose(1, 2).reshape(B, D, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE)
    return feat.squeeze(0).to(torch.float16).cpu().numpy()


def _dino_ok(img_path):
    stem = img_path.name.replace("_leftImg8bit.png", "")
    p = DINOV2_CACHE_DIR / f"{stem}.npz"
    if not p.exists():
        return False
    try:
        with np.load(p) as d:
            return d["features"].shape == (DINOV2_EMBED_DIM, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE)
    except Exception:
        return False


def build_dinov2_cache(internal_train_images, internal_val_images):
    """Each image -> forward frozen DINOv2 -> save features [384, 37, 37] to disk. The model is
    unloaded from GPU afterward (frees memory for UNet training), matching the notebook."""
    global dinov2_model

    DINOV2_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_dino_targets = internal_train_images + internal_val_images
    print(f"Building DINOv2 cache for {len(all_dino_targets)} images (train+val)...")

    dinov2_model = load_dinov2_model()

    n_cached = 0
    n_new = 0
    errors = []

    for img_path in tqdm(all_dino_targets, desc="Build DINOv2 cache"):
        stem = img_path.name.replace("_leftImg8bit.png", "")
        out_path = DINOV2_CACHE_DIR / f"{stem}.npz"
        if out_path.exists():
            try:
                with np.load(out_path) as d:
                    feat_shape = d["features"].shape
                    if feat_shape == (DINOV2_EMBED_DIM, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE):
                        n_cached += 1
                        continue
            except Exception:
                pass

        try:
            image_pil = Image.open(img_path)
            feat = dinov2_forward(image_pil)
            np.savez_compressed(out_path, features=feat)
            n_new += 1
        except Exception as e:
            errors.append((stem, str(e)))

    print(f"\nDINOv2 cache: {n_cached} already existed, {n_new} new, {len(errors)} errors")
    if errors:
        for e in errors[:5]:
            print(f"  {e}")

    del dinov2_model
    dinov2_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("DINOv2 model unloaded from GPU (freed memory for UNet training)")

    internal_train_images = [p for p in internal_train_images if _dino_ok(p)]
    internal_val_images = [p for p in internal_val_images if _dino_ok(p)]
    print(f"Final internal-train: {len(internal_train_images)}   internal-val: {len(internal_val_images)}")

    return internal_train_images, internal_val_images


# ============================================================================
# CE class weights (from GT pixel stats on internal-train)
# ============================================================================

def compute_ce_weights(internal_train_images):
    global CE_WEIGHTS, BOUNDARY_LAMBDA_TENSOR, TVERSKY_TENSOR

    cls_pixels = {c: 0 for c in range(NUM_OUT_CLASSES)}
    per_class_image_count = {c: 0 for c in TARGET_CLASSES}
    missing_gt = []

    for img_path in tqdm(internal_train_images, desc="Scan GT for CE weights"):
        mask_path, is_train_id = get_gt_mask_path(img_path)
        if mask_path is None:
            missing_gt.append(str(img_path))
            continue

        trainid_mask = load_cityscapes_trainid_mask(mask_path, is_train_id)
        valid = trainid_mask != 255
        v_total = int(valid.sum())

        target_total_px = 0
        for cls_name in TARGET_CLASSES:
            tid = TARGET_TO_TRAINID[cls_name]
            local_id = TARGET_TO_LOCAL[cls_name]
            cls_mask = (trainid_mask == tid) & valid
            p = int(cls_mask.sum())
            cls_pixels[local_id] += p
            target_total_px += p
            if p > 0:
                per_class_image_count[cls_name] += 1

        cls_pixels[0] += v_total - target_total_px  # other/bg

    if missing_gt:
        print(f"\u26a0 {len(missing_gt)} images had no matching GT mask found:")
        for m in missing_gt[:5]:
            print(f"  {m}")

    print(f"\nInternal-train images used: {len(internal_train_images)}")
    print("Per-class image presence count:")
    for cls in TARGET_CLASSES:
        print(f"  {cls:10s}: {per_class_image_count[cls]:5d} images")

    print("\nPixel counts (in internal-train set):")
    total = sum(cls_pixels.values())
    for c in range(NUM_OUT_CLASSES):
        name = "other/bg" if c == 0 else LOCAL_TO_TARGET[c]
        pct = 100.0 * cls_pixels[c] / max(total, 1)
        print(f"  ch {c} ({name:10s}): {cls_pixels[c]:>15,d} px  ({pct:5.2f}%)")

    freq = np.array([cls_pixels[c] for c in range(NUM_OUT_CLASSES)], dtype=np.float64)
    inv = 1.0 / np.clip(freq, 1, None)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.2, 5.0)
    inv = inv / inv.mean()

    rider_local = TARGET_TO_LOCAL["rider"]
    inv[rider_local] *= RIDER_CE_BOOST
    inv = inv / inv.mean()

    CE_WEIGHTS = torch.tensor(inv, dtype=torch.float32, device=DEVICE)

    print(f"\nCE class weights (clipped [0.2, 5.0], rider boosted x{RIDER_CE_BOOST}, mean=1):")
    for c in range(NUM_OUT_CLASSES):
        name = "other/bg" if c == 0 else LOCAL_TO_TARGET[c]
        print(f"  ch {c} ({name:10s}): weight = {CE_WEIGHTS[c].item():.3f}")

    BOUNDARY_LAMBDA_TENSOR = torch.tensor(
        [BOUNDARY_LAMBDA[c] for c in TARGET_CLASSES], dtype=torch.float32, device=DEVICE)
    TVERSKY_TENSOR = torch.tensor(
        [TVERSKY_PARAMS[c] for c in TARGET_CLASSES], dtype=torch.float32, device=DEVICE)


# ============================================================================
# Dataset
# ============================================================================

def build_augment_no_hflip():
    return A.Compose([
        A.RandomScale(scale_limit=(-0.25, 0.25), p=0.7),
        A.PadIfNeeded(min_height=IMAGE_H, min_width=IMAGE_W,
                      border_mode=cv2.BORDER_CONSTANT, value=0,
                      mask_value=255, p=1.0),
        A.RandomCrop(height=IMAGE_H, width=IMAGE_W, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.6),
    ])


def compute_boundary_weight(target_np, alpha=1.5):
    gt = target_np.astype(bool)
    if not gt.any() or gt.all():
        return np.ones(target_np.shape, dtype=np.float32)
    d_in = distance_transform_edt(gt)
    d_out = distance_transform_edt(~gt)
    dist = (d_in + d_out).astype(np.float32)
    w = 1.0 / (dist + 1.0) ** alpha
    w_max = w.max()
    return (w / w_max).astype(np.float32) if w_max > 0 else w


class SAM3CityscapesDataset(Dataset):
    """
    Each sample = 1 Cityscapes image (train or internal-val).

    Returns:
      input_base   : [3+NUM_TARGETS, H, W] = RGB(3) + NUM_TARGETS coarse (augmented if aug != None)
      dino_feat    : [384, 37, 37] -- DINOv2 feature of the ORIGINAL image (unaugmented, except HFlip).
      target_label : [H, W] long -- 0..NUM_TARGETS (0 = other/bg, derived directly from GT trainId)
      valid        : [1, H, W] float -- 0 at GT=255 pixels (ignore)
      coarses      : [NUM_TARGETS, H, W] float -- for baseline (coarse) IoU comparison
      bnd_w        : [NUM_TARGETS, H, W] float -- boundary weight map per target class
      image_id     : str
    """

    def __init__(self, image_paths, coarse_dir, dinov2_dir, aug=None):
        self.image_paths = image_paths
        self.coarse_dir = Path(coarse_dir)
        self.dinov2_dir = Path(dinov2_dir)
        self.aug = aug

    def __len__(self):
        return len(self.image_paths)

    def _stem(self, img_path):
        return img_path.name.replace("_leftImg8bit.png", "")

    def _load_coarses(self, stem):
        with np.load(self.coarse_dir / f"{stem}.npz") as d:
            arrs = [d[cls].astype(np.uint8) for cls in TARGET_CLASSES]
        return np.stack(arrs, axis=0)  # (NUM_TARGETS, H, W) at original resolution

    def _load_dinov2(self, stem):
        with np.load(self.dinov2_dir / f"{stem}.npz") as d:
            feat = d["features"].astype(np.float32)  # (384, 37, 37)
        return feat

    def _build_label_map(self, trainid_mask):
        """Derived directly from GT trainId -- NOT cached. label = 0 (other/bg) by default;
        assign local id 1..NUM_TARGETS for each target class."""
        label = np.zeros(trainid_mask.shape, dtype=np.uint8)
        for cls_name in TARGET_CLASSES:
            tid = TARGET_TO_TRAINID[cls_name]
            local_id = TARGET_TO_LOCAL[cls_name]
            label[trainid_mask == tid] = local_id
        return label

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = self._stem(img_path)

        image_np = np.array(Image.open(img_path).convert("RGB"))
        mask_path, is_train_id = get_gt_mask_path(img_path)
        trainid_mask = load_cityscapes_trainid_mask(mask_path, is_train_id)  # (Ho, Wo) 0-18/255
        coarses_np = self._load_coarses(stem)  # (NUM_TARGETS, Ho, Wo) uint8
        dino_feat = self._load_dinov2(stem)  # (384, 37, 37) f32

        # Resize everything to IMAGE_H x IMAGE_W (keeps the original Cityscapes aspect ratio)
        image_np = cv2.resize(image_np, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
        trainid_mask = cv2.resize(trainid_mask, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
        coarses_resized = [
            cv2.resize(coarses_np[c], (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
            for c in range(NUM_TARGETS)
        ]
        coarses_np = np.stack(coarses_resized, axis=0)

        valid_np = (trainid_mask != 255).astype(np.uint8)
        label_np = self._build_label_map(trainid_mask)

        # Augment: RGB + masks (label + valid + NUM_TARGETS coarses). HFlip is coded manually to
        # keep dino_feat in sync.
        masks_to_aug = [label_np, valid_np] + [coarses_np[c] for c in range(NUM_TARGETS)]
        if self.aug is not None:
            if random.random() < 0.5:
                image_np = image_np[:, ::-1].copy()
                for i in range(len(masks_to_aug)):
                    masks_to_aug[i] = masks_to_aug[i][:, ::-1].copy()
                dino_feat = dino_feat[:, :, ::-1].copy()  # flip along width (axis=2)

            out = self.aug(image=image_np, masks=masks_to_aug)
            image_np = out["image"]
            aug_masks = out["masks"]
            label_np, valid_np = aug_masks[0], aug_masks[1]
            coarses_np = np.stack(aug_masks[2:2 + NUM_TARGETS], axis=0)

        label_np = np.asarray(label_np, dtype=np.uint8)
        valid_np = (np.asarray(valid_np) > 0).astype(np.float32)
        coarses_np = (np.asarray(coarses_np) > 0).astype(np.float32)

        bnd_w = np.zeros((NUM_TARGETS, IMAGE_H, IMAGE_W), dtype=np.float32)
        for c in range(NUM_TARGETS):
            local_id = c + 1
            cls_bin = (label_np == local_id).astype(np.uint8)
            bnd_w[c] = compute_boundary_weight(cls_bin)

        rgb = TF.normalize(TF.to_tensor(image_np), IMAGENET_MEAN, IMAGENET_STD)
        coarses_t = torch.from_numpy(coarses_np)
        input_base = torch.cat([rgb, coarses_t], dim=0)  # [17, H, W]

        return {
            "input_base": input_base,
            "dino_feat": torch.from_numpy(dino_feat.astype(np.float32)),
            "target_label": torch.from_numpy(label_np.astype(np.int64)),
            "valid": torch.from_numpy(valid_np).unsqueeze(0),
            "coarses": coarses_t,
            "bnd_w": torch.from_numpy(bnd_w),
            "image_id": stem,
        }


def build_datasets(internal_train_images, internal_val_images):
    train_ds = SAM3CityscapesDataset(internal_train_images, COARSE_CACHE_DIR, DINOV2_CACHE_DIR,
                                      aug=build_augment_no_hflip())
    val_ds = SAM3CityscapesDataset(internal_val_images, COARSE_CACHE_DIR, DINOV2_CACHE_DIR, aug=None)

    print(f"Train samples (internal-train, WITH aug): {len(train_ds)}")
    print(f"Val   samples (internal-val,   NO  aug ): {len(val_ds)}")

    b = train_ds[0]
    assert b["input_base"].shape == (IN_CHANNELS_BASE, IMAGE_H, IMAGE_W)
    assert b["dino_feat"].shape == (DINOV2_EMBED_DIM, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE)
    assert b["target_label"].shape == (IMAGE_H, IMAGE_W)
    assert b["bnd_w"].shape == (NUM_TARGETS, IMAGE_H, IMAGE_W)
    print(f"Sample[0]: image_id={b['image_id']}")
    print(f"  input_base   : {tuple(b['input_base'].shape)}  (RGB+{NUM_TARGETS} coarse)")
    print(f"  dino_feat    : {tuple(b['dino_feat'].shape)}  range [{float(b['dino_feat'].min()):.3f}, {float(b['dino_feat'].max()):.3f}]")
    print(f"  target_label : {tuple(b['target_label'].shape)}  unique: {sorted(b['target_label'].unique().tolist())}")
    print(f"  valid        : {tuple(b['valid'].shape)}  fg={float(b['valid'].mean()):.4f}")
    print(f"  coarses      : {tuple(b['coarses'].shape)}  fg per-class: "
          f"{[round(float(b['coarses'][c].mean()), 4) for c in range(NUM_TARGETS)]}")
    print(f"  bnd_w        : {tuple(b['bnd_w'].shape)}  range: [{float(b['bnd_w'].min()):.3f}, {float(b['bnd_w'].max()):.3f}]")

    return train_ds, val_ds


# ============================================================================
# Model -- ResNet-34 + ASPP + UNet decoder + DINOv2 feature branch
# ============================================================================

class ASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=256):
        super().__init__()

        def _branch(dilation):
            if dilation == 1:
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.b1 = _branch(1)
        self.b6 = _branch(6)
        self.b12 = _branch(12)
        self.b18 = _branch(18)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True))

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([self.b1(x), self.b6(x), self.b12(x), self.b18(x), gap], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ResNet34UNetASPPCityscapes(nn.Module):
    """
    ResNet-34 + ASPP + UNet decoder + DINOv2 feature branch.

    Input:
      - input_base : [B, 3+NUM_TARGETS, H, W]  = RGB(3) + NUM_TARGETS coarse   (H=512, W=1024)
      - dino_feat  : [B, 384, 37, 37]  = DINOv2 ViT-S/14 features (frozen, from cache)

    DINOv2 branch (learned): 1x1 conv 384 -> 32 -> bilinear upsample -> [B, 32, H, W]
    Main path: concat(input_base, dino_compressed) = [B, 3+NUM_TARGETS+32, H, W] -> ResNet first conv.
    H != W, interpolate/transpose-conv both use size=(H,W).

    First conv init:
      - ch 0-2                        : ImageNet pretrained RGB weights
      - ch 3 .. in_channels_base-1    : 0  (NUM_TARGETS coarse channels)
      - ch in_channels_base .. total-1: 0  (DINOv2 compressed, learned 1x1 small random)
    """

    def __init__(self, in_channels_base=17, dinov2_dim=384, dinov2_compress=32, out_channels=15):
        super().__init__()
        self.in_channels_base = in_channels_base
        self.dinov2_dim = dinov2_dim
        self.dinov2_compress = dinov2_compress
        self.total_in = in_channels_base + dinov2_compress  # 49

        self.dino_compress = nn.Sequential(
            nn.Conv2d(dinov2_dim, dinov2_compress, 1, bias=False),
            nn.BatchNorm2d(dinov2_compress),
            nn.ReLU(inplace=True),
        )

        backbone = tvm.resnet34(weights=ResNet34_Weights.DEFAULT)

        orig_w = backbone.conv1.weight.data.clone()  # [64, 3, 7, 7]
        new_conv = nn.Conv2d(self.total_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_w
            new_conv.weight[:, 3:] = 0.0
        backbone.conv1 = new_conv

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.aspp = ASPP(in_ch=512, out_ch=256)

        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, input_base, dino_feat):
        """
        input_base: [B, in_channels_base, H, W]  (H=512, W=1024)
        dino_feat : [B, 384, gh, gw]              (gh=gw=37)
        """
        H, W = input_base.shape[-2:]

        dino_c = self.dino_compress(dino_feat)  # [B, 32, 37, 37]
        dino_u = F.interpolate(dino_c, size=(H, W), mode="bilinear", align_corners=False)

        x = torch.cat([input_base, dino_u], dim=1)  # [B, 49, H, W]

        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottle = self.aspp(e4)
        d = self.dec4(bottle, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.final_up(d)
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(d)  # [B, NUM_OUT_CLASSES, H, W]


def build_model():
    model = ResNet34UNetASPPCityscapes(
        in_channels_base=IN_CHANNELS_BASE,
        dinov2_dim=DINOV2_EMBED_DIM,
        dinov2_compress=DINOV2_COMPRESS,
        out_channels=NUM_OUT_CLASSES,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("ResNet34UNetASPPCityscapes")
    print(f"  input_base channels: {IN_CHANNELS_BASE}  (RGB + {NUM_TARGETS} coarse)")
    print(f"  DINOv2 dim         : {DINOV2_EMBED_DIM} -> compressed {DINOV2_COMPRESS} (1x1 conv learned)")
    print(f"  Total input        : {IN_CHANNELS_TOTAL}  (concat {IN_CHANNELS_BASE} + {DINOV2_COMPRESS})")
    print(f"  out_classes        : {NUM_OUT_CLASSES}")
    print(f"  Image size         : {IMAGE_H} x {IMAGE_W}")
    print(f"  Trainable params   : {n_params:,}")

    with torch.no_grad():
        _x = torch.zeros(1, IN_CHANNELS_BASE, IMAGE_H, IMAGE_W).to(DEVICE)
        _d = torch.zeros(1, DINOV2_EMBED_DIM, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE).to(DEVICE)
        _y = model(_x, _d)
        print(f"  Forward check: input_base={tuple(_x.shape)}, dino={tuple(_d.shape)} -> out={tuple(_y.shape)}  ok")
        assert _y.shape == (1, NUM_OUT_CLASSES, IMAGE_H, IMAGE_W)
    del _x, _d, _y

    return model


# ============================================================================
# Loss: Focal CE + per-class Tversky + Boundary Loss
# ============================================================================

def masked_ce(logits, target_label, valid):
    ce_per_pix = F.cross_entropy(logits, target_label, weight=CE_WEIGHTS, reduction="none")
    if FOCAL_GAMMA > 0:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target_label.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(0.0, 1.0)
        focal_w = (1.0 - pt).pow(FOCAL_GAMMA)
        ce_per_pix = ce_per_pix * focal_w
    v = valid.squeeze(1)
    return (ce_per_pix * v).sum() / v.sum().clamp_min(1.0)


def per_class_loss(probs, target_label, valid, bnd_w):
    v = valid.squeeze(1).float()
    eps = 1.0
    dice_sum = torch.tensor(0.0, device=probs.device)
    bnd_sum = torch.tensor(0.0, device=probs.device)

    for c_idx in range(NUM_TARGETS):
        local_id = c_idx + 1
        p = probs[:, local_id] * v
        t = (target_label == local_id).float() * v

        alpha = TVERSKY_TENSOR[c_idx, 0]
        beta = TVERSKY_TENSOR[c_idx, 1]
        tp = (p * t).sum(dim=(1, 2))
        fp = (p * (1 - t)).sum(dim=(1, 2))
        fn = ((1 - p) * t).sum(dim=(1, 2))
        tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        cls_loss = (1.0 - tv).mean()
        dice_sum = dice_sum + cls_loss

        p_c = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        bce_pix = -(t * torch.log(p_c) + (1.0 - t) * torch.log(1.0 - p_c))
        wmap = bnd_w[:, c_idx] * v
        bnd_c = (bce_pix * wmap).sum() / wmap.sum().clamp_min(1.0)
        bnd_sum = bnd_sum + BOUNDARY_LAMBDA_TENSOR[c_idx] * bnd_c

    dice_avg = dice_sum / NUM_TARGETS
    bnd_avg = bnd_sum / NUM_TARGETS
    return dice_avg, bnd_avg


def combined_loss(logits, target_label, valid, bnd_w):
    ce = masked_ce(logits, target_label, valid)
    probs = F.softmax(logits, dim=1)
    dice_loss, bnd_loss = per_class_loss(probs, target_label, valid, bnd_w)
    total = 0.5 * ce + 0.3 * dice_loss + 0.2 * bnd_loss
    return total, ce.detach(), dice_loss.detach(), bnd_loss.detach()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def init_metric_dict():
    return {cls: {"tp": 0, "fp": 0, "fn": 0, "c_tp": 0, "c_fp": 0, "c_fn": 0} for cls in TARGET_CLASSES}


def update_metrics_multi(mdict, pred_label, target_label, valid_bool, coarses_bool):
    for c_idx, cls in enumerate(TARGET_CLASSES):
        local_id = c_idx + 1
        v = valid_bool
        p = (pred_label == local_id) & v
        t = (target_label == local_id) & v
        c = coarses_bool[:, c_idx] & v
        mdict[cls]["tp"] += int((p & t).sum().item())
        mdict[cls]["fp"] += int((p & ~t).sum().item())
        mdict[cls]["fn"] += int((~p & t).sum().item())
        mdict[cls]["c_tp"] += int((c & t).sum().item())
        mdict[cls]["c_fp"] += int((c & ~t).sum().item())
        mdict[cls]["c_fn"] += int((~c & t).sum().item())


def iou_from(tp, fp, fn):
    d = tp + fp + fn
    return float(tp / d) if d > 0 else float("nan")


def summarize(mdict):
    per_iou = {cls: iou_from(mdict[cls]["tp"], mdict[cls]["fp"], mdict[cls]["fn"]) for cls in TARGET_CLASSES}
    per_coarse = {cls: iou_from(mdict[cls]["c_tp"], mdict[cls]["c_fp"], mdict[cls]["c_fn"]) for cls in TARGET_CLASSES}
    miou = float(np.nanmean(list(per_iou.values())))
    coarse_miou = float(np.nanmean(list(per_coarse.values())))
    return {"per_iou": per_iou, "per_coarse": per_coarse, "miou": miou, "coarse_miou": coarse_miou,
            "delta": miou - coarse_miou}


# ============================================================================
# DataLoader + Optimizer + Warmup-Cosine LR scheduler
# ============================================================================

def build_dataloaders_and_optimizer(train_ds, val_ds, model):
    pin = DEVICE.type == "cuda"
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin, drop_last=False)
    if NUM_WORKERS > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    print(f"Train batches: {len(train_loader)}  ({len(train_ds)} images, WITH aug)")
    print(f"Val   batches: {len(val_loader)}    ({len(val_ds)} images, NO  aug -- real internal-val)")

    _b = next(iter(train_loader))
    print(f"Batch input_base   : {tuple(_b['input_base'].shape)}    expect [B,{IN_CHANNELS_BASE},{IMAGE_H},{IMAGE_W}]")
    print(f"Batch dino_feat    : {tuple(_b['dino_feat'].shape)}    expect [B,{DINOV2_EMBED_DIM},{DINOV2_GRID_SIZE},{DINOV2_GRID_SIZE}]")
    print(f"Batch target_label : {tuple(_b['target_label'].shape)}  expect [B,{IMAGE_H},{IMAGE_W}]")
    print(f"Batch bnd_w        : {tuple(_b['bnd_w'].shape)}        expect [B,{NUM_TARGETS},{IMAGE_H},{IMAGE_W}]")
    assert _b["input_base"].shape[1] == IN_CHANNELS_BASE
    assert _b["dino_feat"].shape[1] == DINOV2_EMBED_DIM
    del _b

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast_ctx():
        return torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    print(f"Optimizer : AdamW  lr={LR}  wd={WEIGHT_DECAY}")
    print(f"Scheduler : Linear warmup ({WARMUP_EPOCHS} ep) -> CosineAnnealing -> 0.05*LR")
    print(f"AMP       : {use_amp}")

    return train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx


# ============================================================================
# Training loop -- early-stop on internal-val mIoU
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_miou = -1.0
    best_epoch = -1
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_ce = tr_dice = tr_bnd = 0.0
        tr_n = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [train]", leave=False)
        for batch in bar:
            inp_base = batch["input_base"].to(DEVICE, non_blocking=True)
            dino_feat = batch["dino_feat"].to(DEVICE, non_blocking=True)
            tgt_label = batch["target_label"].to(DEVICE, non_blocking=True)
            valid = batch["valid"].to(DEVICE, non_blocking=True)
            bnd_w = batch["bnd_w"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(inp_base, dino_feat)
                loss, ce_d, dice_d, bnd_d = combined_loss(logits, tgt_label, valid, bnd_w)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = inp_base.size(0)
            tr_loss += loss.item() * bs
            tr_ce += ce_d.item() * bs
            tr_dice += dice_d.item() * bs
            tr_bnd += bnd_d.item() * bs
            tr_n += bs

            bar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce_d.item():.4f}",
                             dice=f"{dice_d.item():.4f}", bnd=f"{bnd_d.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # ── Eval on real internal-val (not train self-eval) ─────────────────────
        model.eval()
        mdict = init_metric_dict()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [val  ]", leave=False):
                inp_base = batch["input_base"].to(DEVICE, non_blocking=True)
                dino_feat = batch["dino_feat"].to(DEVICE, non_blocking=True)
                tgt_label = batch["target_label"].to(DEVICE, non_blocking=True)
                valid = batch["valid"].to(DEVICE, non_blocking=True)
                coarses = batch["coarses"].to(DEVICE, non_blocking=True)
                with autocast_ctx():
                    logits = model(inp_base, dino_feat)
                pred_label = logits.argmax(dim=1)
                valid_bool = valid.squeeze(1) >= 0.5
                coarses_bool = coarses >= 0.5
                update_metrics_multi(mdict, pred_label, tgt_label, valid_bool, coarses_bool)

        n = max(tr_n, 1)
        s = summarize(mdict)
        rec = dict(epoch=epoch, train_loss=tr_loss / n, ce=tr_ce / n, dice=tr_dice / n, bnd=tr_bnd / n,
                   miou=s["miou"], coarse_miou=s["coarse_miou"], delta=s["delta"],
                   lr=float(optimizer.param_groups[0]["lr"]),
                   per_iou={k: float(v) for k, v in s["per_iou"].items()},
                   per_coarse={k: float(v) for k, v in s["per_coarse"].items()})
        history.append(rec)

        print(f"\nEpoch {epoch:2d}/{EPOCHS}")
        print(f"  Loss: {tr_loss/n:.4f}  (ce={tr_ce/n:.4f}  dice={tr_dice/n:.4f}  bnd={tr_bnd/n:.4f})")
        print(f"  [internal-val] Coarse mIoU: {s['coarse_miou']:.4f}  ->  Refined mIoU: {s['miou']:.4f}  (delta {s['delta']:+.4f})")
        print("  Per-class (coarse -> refined), internal-val:")
        for cls in TARGET_CLASSES:
            ci = s["per_coarse"][cls]
            ri = s["per_iou"][cls]
            print(f"    {cls:10s}: {ci:.4f} -> {ri:.4f}  (delta {ri-ci:+.4f})")

        ckpt = dict(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            miou=s["miou"],
            coarse_miou=s["coarse_miou"],
            target_classes=TARGET_CLASSES,
            target_to_trainid=TARGET_TO_TRAINID,
            target_to_local=TARGET_TO_LOCAL,
            boundary_lambda=BOUNDARY_LAMBDA,
            in_channels_base=IN_CHANNELS_BASE,
            in_channels_total=IN_CHANNELS_TOTAL,
            dinov2_model_name=DINOV2_MODEL_NAME,
            dinov2_embed_dim=DINOV2_EMBED_DIM,
            dinov2_patch_size=DINOV2_PATCH_SIZE,
            dinov2_input_size=DINOV2_INPUT_SIZE,
            dinov2_grid_size=DINOV2_GRID_SIZE,
            dinov2_compress=DINOV2_COMPRESS,
            num_out_classes=NUM_OUT_CLASSES,
            image_h=IMAGE_H,
            image_w=IMAGE_W,
            ce_weights=CE_WEIGHTS.detach().cpu().numpy().tolist(),
            tversky_params=TVERSKY_PARAMS,
            rider_ce_boost=RIDER_CE_BOOST,
            focal_gamma=FOCAL_GAMMA,
            arch_version="multi_class_softmax_city_v1_dinov2",
        )
        torch.save(ckpt, LAST_CKPT)

        if s["miou"] > best_miou:
            best_miou = s["miou"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(ckpt, BEST_CKPT)
            print(f"  BEST checkpoint (epoch {epoch}  mIoU={best_miou:.4f})")
        else:
            epochs_since_best += 1
            print(f"  no improvement ({epochs_since_best}/{PATIENCE} since best epoch {best_epoch}, mIoU={best_miou:.4f})")

        with open(HISTORY_JSON, "w") as f:
            json.dump(history, f, indent=2)

        if epochs_since_best >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no internal-val mIoU improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training done | Best epoch: {best_epoch} | Best internal-val mIoU: {best_miou:.4f}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"Best ckpt : {BEST_CKPT}")
    print(f"Last ckpt : {LAST_CKPT}")

    return history, best_epoch, best_miou


# ============================================================================
# Plot training curves (saved to PNG -- this VM has no display)
# ============================================================================

def plot_curves(history, best_epoch, best_miou, out_path):
    ep = [r["epoch"] for r in history]
    tr_loss = [r["train_loss"] for r in history]
    ce_v = [r["ce"] for r in history]
    dice_v = [r["dice"] for r in history]
    bnd_v = [r["bnd"] for r in history]
    miou = [r["miou"] for r in history]
    cmiou = [r["coarse_miou"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ep, tr_loss, "o-", label="Total")
    axes[0].plot(ep, ce_v, "s--", label="Focal CE", alpha=0.7)
    axes[0].plot(ep, dice_v, "^--", label="Tversky/Dice", alpha=0.7)
    axes[0].plot(ep, bnd_v, "d--", label="Boundary", alpha=0.7)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training Losses")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, cmiou, "o-", label="Coarse (SAM3)", color="#d9534f")
    axes[1].plot(ep, miou, "s-", label="Refined (UNet+ASPP+DINOv2)", color="#2ca02c")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].set_title(f"Internal-val mIoU ({NUM_TARGETS}-class mean)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    for cls in TARGET_CLASSES:
        pi = [r["per_iou"][cls] for r in history]
        axes[2].plot(ep, pi, "o-", label=cls, alpha=0.8)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("IoU")
    axes[2].set_title("Per-class IoU (refined, internal-val)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.suptitle(f"UNet+ASPP+DINOv2 Cityscapes -- Best Epoch {best_epoch} | Internal-val mIoU {best_miou:.4f}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    setup_torch()
    set_seed(SEED)

    internal_train_images, internal_val_images = load_and_validate_split()
    internal_train_images, internal_val_images = build_dinov2_cache(internal_train_images, internal_val_images)
    compute_ce_weights(internal_train_images)
    train_ds, val_ds = build_datasets(internal_train_images, internal_val_images)
    model = build_model()
    train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx = \
        build_dataloaders_and_optimizer(train_ds, val_ds, model)

    history, best_epoch, best_miou = train_model(
        model, train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx)

    if history:
        plot_curves(history, best_epoch, best_miou, PLOT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] train_unetasppdinov2_city_llm.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
