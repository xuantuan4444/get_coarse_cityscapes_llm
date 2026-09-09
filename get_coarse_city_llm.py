import argparse
import gc
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# CONFIG — every path is resolved from PROJECT_ROOT (this script's own folder), never from the
# current working directory and never from an environment variable, so it can't silently point
# at the wrong place depending on where/how the script is launched.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

CITYSCAPES_ROOT = PROJECT_ROOT / "data_cityscapes"
CITYSCAPES_TRAIN_ROOT = CITYSCAPES_ROOT / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train"
CITYSCAPES_GT_ROOT = CITYSCAPES_ROOT / "gtFine_trainvaltest" / "gtFine" / "train"

SAM3_CKPT = PROJECT_ROOT / "weight_sam3" / "sam3.pt"
ADJUST_PROMPT_PATH = PROJECT_ROOT / "adjust_prompt_city_v3.json"

# Output: coarse mask cache + its zip, named for this arm (city / LLM) so it never collides with
# the no-LLM or VOC cache directories if those are ever run from a sibling project folder.
COARSE_CACHE_DIR = PROJECT_ROOT / "coarse_cache_city_llm"
COARSE_CACHE_ZIP = PROJECT_ROOT / "coarse_cache_city_llm.zip"

COARSE_THRESHOLDS = [0.50, 0.30, 0.20, 0.15]
SEED = 42
INTERNAL_VAL_FRACTION = 0.10
GC_EVERY_N_IMAGES = 64  # gc.collect()/empty_cache() cadence during cache build, same as notebook

CITYSCAPES_CLASSES_ALL = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
GT_CLASS_TO_TRAINID = {name: idx for idx, name in enumerate(CITYSCAPES_CLASSES_ALL)}

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


# ============================================================================
# Environment setup
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        # Entered once, never exited — matches the notebook's global-autocast pattern for the
        # rest of the process. Guarded here in case this script ever runs on a CPU-only VM.
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device  :", device)
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))
    return device


# ============================================================================
# Prompt ensemble / target classes
# ============================================================================

def load_prompt_ensemble(adjust_prompt_path: Path):
    """Load the LLM-generated prompt ensemble; TARGET_CLASSES = the keys of adjust_prompt.json."""
    if adjust_prompt_path.exists():
        with open(adjust_prompt_path, encoding="utf-8") as f:
            adjust_prompt = json.load(f)
    else:
        print(f"[warn] {adjust_prompt_path} not found -- every class falls back to its own name as the single prompt.")
        adjust_prompt = {}

    target_classes = list(adjust_prompt.keys())
    class_prompts = [adjust_prompt[c] for c in target_classes]
    return target_classes, class_prompts


# ============================================================================
# Image discovery & internal train/val split
# ============================================================================

def discover_cityscapes_images(img_root: Path):
    paths = []
    for city_dir in sorted(img_root.glob("*")):
        if not city_dir.is_dir():
            continue
        for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            paths.append(img_path)
    return paths


def split_internal_train_val(all_images, cache_dir: Path, seed: int = SEED,
                              val_fraction: float = INTERNAL_VAL_FRACTION):
    """Carve an internal train/val split out of the official train images (~10% val)."""
    rng = random.Random(seed)
    shuffled = all_images.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    internal_val = shuffled[:n_val]
    internal_train = shuffled[n_val:]

    split_path = cache_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump({
            "train": [str(p) for p in internal_train],
            "val": [str(p) for p in internal_val],
        }, f)
    print(f"Internal train: {len(internal_train)} | Internal val: {len(internal_val)}")
    print(f"Split saved to {split_path}")
    return internal_train, internal_val


# ============================================================================
# GT mask helpers
# ============================================================================

def get_gt_mask_path(img_path: Path, gt_root: Path):
    """.../leftImg8bit/train/<city>/<city>_..._leftImg8bit.png
       -> .../gtFine/train/<city>/<city>_..._gtFine_labelIds.png (or _labelTrainIds.png)"""
    city = img_path.parent.name
    stem = img_path.name.replace("_leftImg8bit.png", "")
    mask_dir = gt_root / city
    train_id_path = mask_dir / f"{stem}_gtFine_labelTrainIds.png"
    label_id_path = mask_dir / f"{stem}_gtFine_labelIds.png"
    if train_id_path.exists():
        return train_id_path, True
    elif label_id_path.exists():
        return label_id_path, False
    return None, None


def load_cityscapes_trainid_mask(mask_path: Path, is_train_id: bool):
    """Raw trainId map (0-18, 255=ignore) — used only for per-class pixel stats, not shifted."""
    raw = np.array(Image.open(mask_path), dtype=np.uint8)
    return raw if is_train_id else _ID2TRAIN_LUT[raw]


# ============================================================================
# SAM3 coarse cache precomputation
# ============================================================================

def _extract_masks_scores(state):
    if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
        return [], []
    masks, scores = [], []
    for i in range(len(state["masks"])):
        masks.append(state["masks"][i].squeeze(0).detach().cpu().numpy().astype(bool))
        scores.append(float(state["scores"][i].item()))
    return masks, scores


def _union_at_threshold(masks, scores, shape_hw, thresholds):
    h, w = shape_hw
    scores_np = np.asarray(scores, dtype=np.float32)
    for thr in thresholds:
        idx = np.where(scores_np >= thr)[0]
        if len(idx) > 0:
            union = np.zeros((h, w), dtype=bool)
            for i in idx:
                union |= masks[i]
            return union.astype(np.uint8), thr
    return np.zeros((h, w), dtype=np.uint8), None


def _safe_cuda_empty_cache():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[warn] torch.cuda.empty_cache failed: {e}")


def build_sam3_processor(sam3_ckpt: Path, bpe_path: Path, coarse_thresholds):
    if not sam3_ckpt.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {sam3_ckpt}")

    model_sam = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(sam3_ckpt),
        load_from_HF=False,
    )

    if torch.cuda.is_available():
        try:
            model_sam = model_sam.cuda()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[warn] CUDA OOM while moving SAM3 to GPU; fallback to CPU for cache build.")
                _safe_cuda_empty_cache()
            else:
                raise

    conf_min = min(coarse_thresholds)
    processor = Sam3Processor(model_sam, confidence_threshold=conf_min)
    return model_sam, processor


def build_coarse_cache(image_paths, target_classes, class_prompts, coarse_cache_dir: Path,
                        coarse_thresholds, sam3_ckpt: Path, bpe_path: Path,
                        force_rebuild=False, processor=None, gc_every=GC_EVERY_N_IMAGES):
    """
    Runs SAM3 once, saves a binary (0/1) union mask per target class into .npz.
    Each file: {stem}.npz with keys = target_classes, value = (H,W) uint8 mask.
    """
    missing = [
        p for p in image_paths
        if force_rebuild or not (coarse_cache_dir / f"{p.name.replace('_leftImg8bit.png', '')}.npz").exists()
    ]
    if not missing:
        print(f"Coarse cache complete ({len(image_paths)} images). Skipping.")
        return []

    print(f"Building coarse cache: {len(missing)}/{len(image_paths)} images...")
    own_processor = processor is None
    model_sam = None

    if own_processor:
        model_sam, processor = build_sam3_processor(sam3_ckpt, bpe_path, coarse_thresholds)

    errors = []
    try:
        for step, img_path in enumerate(tqdm(missing, desc="SAM3 coarse cache (Cityscapes)"), start=1):
            stem = img_path.name.replace("_leftImg8bit.png", "")
            cache_path = coarse_cache_dir / f"{stem}.npz"
            try:
                image = Image.open(img_path).convert("RGB")
                state = processor.set_image(image)

                coarse_dict = {}
                for cls_name, prompts in zip(target_classes, class_prompts):
                    all_masks, all_scores = [], []
                    for prompt in prompts:
                        processor.reset_all_prompts(state)
                        state = processor.set_text_prompt(state=state, prompt=prompt)
                        masks, scores = _extract_masks_scores(state)
                        all_masks.extend(masks)
                        all_scores.extend(scores)

                    union, _ = _union_at_threshold(
                        all_masks, all_scores, (image.height, image.width), coarse_thresholds
                    )
                    coarse_dict[cls_name] = union

                np.savez_compressed(cache_path, **coarse_dict)
            except Exception as e:
                errors.append((stem, str(e)))

            if step % gc_every == 0:
                gc.collect()
                _safe_cuda_empty_cache()
    finally:
        if own_processor:
            try:
                del processor
            except Exception:
                pass
            if model_sam is not None:
                del model_sam
            gc.collect()
            _safe_cuda_empty_cache()

    print(f"Done. Errors: {len(errors)}")
    for stem, err in errors[:5]:
        print(f"  {stem}: {err}")
    return errors


def audit_and_rebuild_cache(all_images, target_classes, class_prompts, coarse_cache_dir: Path,
                             coarse_thresholds, sam3_ckpt: Path, bpe_path: Path):
    """Audit cache & rebuild only what is missing/corrupt (avoids reloading SAM3 repeatedly)."""
    req_keys = set(target_classes)

    def _cache_ok(stem):
        p = coarse_cache_dir / f"{stem}.npz"
        if not p.exists():
            return False, "missing_file"
        try:
            with np.load(p) as d:
                keys = set(d.files)
                miss = req_keys - keys
                if miss:
                    return False, f"missing_keys:{sorted(miss)}"
                for k in req_keys:
                    arr = d[k]
                    if arr.ndim != 2:
                        return False, f"bad_ndim:{k}:{arr.ndim}"
                return True, "ok"
        except Exception as e:
            return False, f"corrupt:{type(e).__name__}:{str(e)[:120]}"

    stem_to_path = {p.name.replace("_leftImg8bit.png", ""): p for p in all_images}
    all_stems = list(stem_to_path.keys())

    todo_stems = []
    todo_reason = {}
    for stem in tqdm(all_stems, desc="Audit coarse_cache_city"):
        ok, reason = _cache_ok(stem)
        if not ok:
            todo_stems.append(stem)
            todo_reason[stem] = reason

    print(f"Need rebuild: {len(todo_stems)}/{len(all_stems)}")
    for stem in todo_stems[:10]:
        print(f"  {stem}: {todo_reason[stem]}")

    if len(todo_stems) == 0:
        print("Cache is complete, no need to rebuild.")
    else:
        print("Rebuilding missing/corrupt cache in one pass (single SAM3 load)...")
        build_coarse_cache(
            [stem_to_path[s] for s in todo_stems], target_classes, class_prompts,
            coarse_cache_dir, coarse_thresholds, sam3_ckpt, bpe_path, force_rebuild=True,
        )

    remain = [s for s in all_stems if not _cache_ok(s)[0]]
    print(f"Cache ready: {len(all_stems) - len(remain)}/{len(all_stems)}")
    print(f"Still missing/corrupt: {len(remain)}")
    print("Sample remain:", remain[:10])
    return remain


# ============================================================================
# Per-class GT pixel stats (for CE class weights downstream)
# ============================================================================

def scan_gt_class_stats(internal_train_images, target_classes, gt_root: Path, coarse_cache_dir: Path):
    cls_pos_px = {c: 0 for c in target_classes}
    cls_neg_px = {c: 0 for c in target_classes}
    train_pos_stems = {c: set() for c in target_classes}
    missing_gt = []

    for img_path in tqdm(internal_train_images, desc="Scanning GT for per-class pixel stats"):
        mask_path, is_train_id = get_gt_mask_path(img_path, gt_root)
        if mask_path is None:
            missing_gt.append(str(img_path))
            continue

        trainid_mask = load_cityscapes_trainid_mask(mask_path, is_train_id)
        valid = trainid_mask != 255
        total_valid = int(valid.sum())
        stem = img_path.name.replace("_leftImg8bit.png", "")

        for cls_name in target_classes:
            tid = GT_CLASS_TO_TRAINID[cls_name]
            cls_mask = (trainid_mask == tid) & valid
            p = int(cls_mask.sum())
            if p > 0:
                train_pos_stems[cls_name].add(stem)
                cls_pos_px[cls_name] += p
                cls_neg_px[cls_name] += (total_valid - p)

    if missing_gt:
        print(f"[warn] {len(missing_gt)} images had no matching GT mask found (check CITYSCAPES_GT_ROOT):")
        for m in missing_gt[:5]:
            print(f"  {m}")

    # Per-class weight: neg/pos ratio, clipped [1, 50]
    cls_pos_weights = {}
    for cls_name in target_classes:
        ratio = cls_neg_px[cls_name] / max(cls_pos_px[cls_name], 1)
        cls_pos_weights[cls_name] = float(np.clip(ratio, 1.0, 50.0))

    print(f"\nPer-class stats (internal train images, n={len(internal_train_images)}):")
    for cls_name in target_classes:
        print(f"  {cls_name:12s} | pos_images={len(train_pos_stems[cls_name]):5d} "
              f"| pos_px={cls_pos_px[cls_name]:>13,} | neg_px={cls_neg_px[cls_name]:>15,} "
              f"| pos_weight={cls_pos_weights[cls_name]:.2f}")

    stats_path = coarse_cache_dir / "class_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "cls_pos_px": cls_pos_px,
            "cls_neg_px": cls_neg_px,
            "cls_pos_weights": cls_pos_weights,
            "train_pos_stems": {k: sorted(v) for k, v in train_pos_stems.items()},
        }, f, indent=1)
    print(f"\nSaved to {stats_path}")


# ============================================================================
# Packaging & sanity-check visualization
# ============================================================================

def zip_coarse_cache(coarse_cache_dir: Path, zip_path: Path):
    """Zip the full cache directory (npz masks + split.json + class_stats.json) — the final
    deliverable to download from this VM."""
    if not coarse_cache_dir.exists():
        print(f"[warn] Cache directory not found: {coarse_cache_dir}")
        return

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=str(coarse_cache_dir.parent),
        base_dir=coarse_cache_dir.name,
    )
    print(f"ZIP created: {zip_path}")
    print(f"Cache files: {len(list(coarse_cache_dir.glob('*.npz')))}")


def visualize_sample(internal_train_images, target_classes, coarse_cache_dir: Path, out_path: Path):
    """Save (not show — this VM has no display) a per-class mask grid for one sample image,
    as a quick visual sanity check of the produced cache."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe backend, no X server needed on a remote VM
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed -- skipping sample visualization.")
        return

    sample_stems = [p.name.replace("_leftImg8bit.png", "") for p in internal_train_images[:1]]
    for stem in sample_stems:
        cache_path = coarse_cache_dir / f"{stem}.npz"
        if not cache_path.exists():
            print(f"[skip] no cache for {stem} yet")
            continue

        with np.load(cache_path) as d:
            channels = target_classes
            n = len(channels)
            fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
            axes = axes.flatten()
            for ax, ch in zip(axes, channels):
                ax.imshow(d[ch], vmin=0, vmax=1, cmap="gray")
                ax.set_title(ch, fontsize=10)
                ax.axis("off")
            for ax in axes[len(channels):]:
                ax.axis("off")
            fig.suptitle(f"Coarse binary masks — {stem}", fontsize=13)
            plt.tight_layout()
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"Sample visualization saved to {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute SAM3 coarse masks for Cityscapes (LLM prompt ensemble).")
    parser.add_argument(
        "--rebuild-cache", action="store_true", default=False,
        help="Force-rebuild every cache entry instead of only the missing ones (default: off).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    setup_torch()
    set_seed(SEED)

    COARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    target_classes, class_prompts = load_prompt_ensemble(ADJUST_PROMPT_PATH)
    print(f"Target classes ({len(target_classes)}): {target_classes}")
    print(f"Prompt ensembles: {class_prompts}")
    print(f"Cache dir: {COARSE_CACHE_DIR}")
    print(f"SAM3 ckpt: {SAM3_CKPT}")

    sam3_dir = Path(sam3.__file__).parent
    bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    all_train_images = discover_cityscapes_images(CITYSCAPES_TRAIN_ROOT)
    print(f"Found {len(all_train_images)} images under {CITYSCAPES_TRAIN_ROOT}")
    if not all_train_images:
        raise RuntimeError(
            f"No images found under CITYSCAPES_TRAIN_ROOT={CITYSCAPES_TRAIN_ROOT}. "
            "Check the path."
        )

    internal_train_images, internal_val_images = split_internal_train_val(all_train_images, COARSE_CACHE_DIR)

    # Step 1: build cache for every image missing one (or all, if --rebuild-cache)
    build_coarse_cache(
        all_train_images, target_classes, class_prompts, COARSE_CACHE_DIR, COARSE_THRESHOLDS,
        SAM3_CKPT, bpe_path, force_rebuild=args.rebuild_cache,
    )

    # Step 2: audit the full cache and rebuild anything still missing/corrupt, in one more pass
    audit_and_rebuild_cache(
        all_train_images, target_classes, class_prompts, COARSE_CACHE_DIR, COARSE_THRESHOLDS,
        SAM3_CKPT, bpe_path,
    )

    # Step 3: per-class GT pixel stats on the internal-train split (for downstream CE weights)
    scan_gt_class_stats(internal_train_images, target_classes, CITYSCAPES_GT_ROOT, COARSE_CACHE_DIR)

    # Step 4: package the final deliverable
    zip_coarse_cache(COARSE_CACHE_DIR, COARSE_CACHE_ZIP)

    # Step 5: quick visual sanity check (non-essential, best-effort)
    visualize_sample(internal_train_images, target_classes, COARSE_CACHE_DIR,
                      COARSE_CACHE_DIR.parent / "sample_coarse_masks.png")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed / 60:.1f} min")
    print(f"Deliverable zip: {COARSE_CACHE_ZIP}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] get_coarse_city_llm.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)