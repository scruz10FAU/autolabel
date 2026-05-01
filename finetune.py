"""
Finetune a YOLOv8 model on the local buoy dataset.

Usage:
    python finetune.py [options]

Options are documented in the argparse block below. Defaults work out of the box
with the repo layout (trainImagesZed/ + training_data/, models/best_alex.pt).
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Finetune YOLOv8 on buoy dataset")
    p.add_argument(
        "--model", "-m", default="models/best_alex.pt",
        help="Path to the base .pt checkpoint to finetune (default: models/best_alex.pt)"
    )
    p.add_argument(
        "--data-roots", "-r", nargs="+",
        default=["trainImagesZed"],
        help="Dataset root folders, each must contain images/ and labels/ subdirs"
    )
    p.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of images held out for validation (default: 0.15)"
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)"
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size in pixels (default: 640)"
    )
    p.add_argument(
        "--batch", type=int, default=16,
        help="Batch size; use -1 for AutoBatch (default: 16)"
    )
    p.add_argument(
        "--lr0", type=float, default=0.001,
        help="Initial learning rate (default: 0.001 — lower than scratch to preserve weights)"
    )
    p.add_argument(
        "--freeze", type=int, default=10,
        help="Number of backbone layers to freeze (default: 10). Set 0 to unfreeze all."
    )
    p.add_argument(
        "--device", default="",
        help="Device: '' for auto, '0' for GPU 0, 'cpu', etc."
    )
    p.add_argument(
        "--project", default="runs/finetune",
        help="Output directory for training runs (default: runs/finetune)"
    )
    p.add_argument(
        "--name", "-n", default="buoy",
        help="Run name inside --project (default: buoy)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val split (default: 42)"
    )
    p.add_argument(
        "--rebuild-split", action="store_true",
        help="Rebuild autosplit_train/ and autosplit_val/ even if they already exist"
    )
    p.add_argument(
        "--classes", '-c', default="classes.json",
        help="JSON file mapping class IDs to names (default: classes.json)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def load_classes(classes_path: Path) -> dict[int, str]:
    data = json.loads(classes_path.read_text())
    return {int(k): v["name"] for k, v in data.items()}


def build_split(data_roots: list[str], val_fraction: float, seed: int,
                rebuild: bool, base: Path, classes: dict[int, str]):
    """
    Collect all (image, label) pairs from *data_roots*, shuffle, split into
    train/val, and copy them into autosplit_train/ and autosplit_val/.
    Returns the path to the generated dataset YAML.
    """
    train_img = base / "autosplit_train" / "images"
    train_lbl = base / "autosplit_train" / "labels"
    val_img   = base / "autosplit_val"   / "images"
    val_lbl   = base / "autosplit_val"   / "labels"

    if not rebuild and train_img.exists() and val_img.exists():
        print("[split] Using existing autosplit_train/ and autosplit_val/")
        return _write_yaml(base, train_img, val_img, classes)

    for d in (train_img, train_lbl, val_img, val_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    pairs: list[tuple[Path, Path]] = []
    for root_str in data_roots:
        root = base / root_str
        img_dir = root / "images"
        lbl_dir = root / "labels"
        if not img_dir.exists():
            print(f"[split] WARNING: {img_dir} not found — skipping")
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                print(f"[split] WARNING: no label for {img.name} — skipping")
                continue
            pairs.append((img, lbl))

    if not pairs:
        raise RuntimeError("No valid image/label pairs found. Check --data-roots.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    print(f"[split] {len(train_pairs)} train / {len(val_pairs)} val "
          f"(total {len(pairs)}, val_fraction={val_fraction})")

    def _copy(pairs, img_dst, lbl_dst):
        for img, lbl in pairs:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(lbl, lbl_dst / lbl.name)

    _copy(train_pairs, train_img, train_lbl)
    _copy(val_pairs,   val_img,   val_lbl)

    return _write_yaml(base, train_img, val_img, classes)


def _write_yaml(base: Path, train_img: Path, val_img: Path,
                classes: dict[int, str]) -> Path:
    names = "\n".join(f"  {i}: {classes[i]}" for i in sorted(classes))
    yaml_path = base / "autosplit_dataset.yaml"
    yaml_path.write_text(
        f"path: {base.resolve()}\n"
        f"train: {train_img.relative_to(base)}\n"
        f"val:   {val_img.relative_to(base)}\n"
        f"\n"
        f"nc: {len(classes)}\n"
        f"names:\n"
        f"{names}\n"
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(args):
    base = Path(__file__).parent.resolve()

    classes_path = base / args.classes
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    classes = load_classes(classes_path)
    print(f"[classes] Loaded {len(classes)} classes from {classes_path.name}: "
          + ", ".join(f"{i}={n}" for i, n in sorted(classes.items())))

    yaml_path = build_split(
        args.data_roots,
        args.val_split,
        args.seed,
        args.rebuild_split,
        base,
        classes,
    )

    model_path = base / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n[train] Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"[train] Starting finetuning for {args.epochs} epochs …\n")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=0.01,           # final LR = lr0 * lrf
        freeze=args.freeze,
        device=args.device if args.device else None,
        project=str(base / args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        weight_decay=0.0005,
        warmup_epochs=3,
        val=True,
        save=True,
        plots=True,
        seed=args.seed,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out  = base / "models" / "finetuned.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    print(f"\n[done] Best weights copied to {out}")
    print(f"[done] Full run saved in {results.save_dir}")


if __name__ == "__main__":
    finetune(parse_args())
