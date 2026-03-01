import argparse
import random
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def geometric_rotate(image, mask, angle):
    image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
    mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)
    return image, mask


def geometric_translate(image, mask, dx, dy):
    image = TF.affine(
        image,
        angle=0.0,
        translate=[dx, dy],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    mask = TF.affine(
        mask,
        angle=0.0,
        translate=[dx, dy],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0,
    )
    return image, mask


def build_augmentations():
    return [
        # ("hflip", lambda image, mask: (TF.hflip(image), TF.hflip(mask))),
        ("rot_p15", lambda image, mask: geometric_rotate(image, mask, angle=15.0)),
        ("rot_m15", lambda image, mask: geometric_rotate(image, mask, angle=-15.0)),
        ("translate_x20", lambda image, mask: geometric_translate(image, mask, dx=20, dy=0)),
        # ("brightness_up", lambda image, mask: (TF.adjust_brightness(image, 1.2), mask)),
        ("contrast_up", lambda image, mask: (TF.adjust_contrast(image, 1.2), mask)),
    ]


def select_augmentation_for_index(index: int, augmentations):
    if not augmentations:
        raise ValueError("No augmentation methods configured.")
    return augmentations[index % len(augmentations)]


def save_mask(mask: Image.Image, save_path: Path):
    mask_array = np.array(mask, dtype=np.uint8)
    Image.fromarray(mask_array, mode="L").save(save_path)


def make_debug_plot(records, output_path: Path, num_samples: int = 10, seed: int = 42):
    if not records:
        return None

    random.seed(seed)
    sample_count = min(num_samples, len(records))
    sampled = random.sample(records, sample_count)

    rows = sample_count
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, max(4, 2.8 * rows)))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(sample_count):
        rec = sampled[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        mask = Image.open(rec["mask_path"])

        image_ax = axes[idx, 0]
        mask_ax = axes[idx, 1]

        image_ax.imshow(image)
        image_ax.set_title(f"{rec['method']}\n{rec['source']}", fontsize=9)
        image_ax.axis("off")

        mask_ax.imshow(mask, cmap="nipy_spectral", interpolation="nearest")
        mask_ax.set_title("Mask", fontsize=9)
        mask_ax.axis("off")

    fig.suptitle("Sampled Augmented Image-Mask Pairs", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate offline data augmentation outputs for CelebAMask training data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--num-plot-samples",
        type=int,
        default=10,
        help="Number of augmented samples to include in debug plot",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    train_images_dir = (project_root / config["data"]["train"]["images"]).resolve()
    train_masks_dir = (project_root / config["data"]["train"]["masks"]).resolve()

    train_aug_root = project_root / "data" / "train_aug"
    out_images_dir = train_aug_root / "images"
    out_masks_dir = train_aug_root / "masks"
    ensure_dir(out_images_dir)
    ensure_dir(out_masks_dir)

    output_dir = (project_root / config["output"]["dir"]).resolve()
    ensure_dir(output_dir)
    plot_output_path = output_dir / "augmentation_debug_samples.png"

    image_files = sorted(train_images_dir.glob("*.jpg"))
    augmentations = build_augmentations()

    debug_records = []
    processed = 0
    skipped = 0
    method_counts = Counter()

    for image_path in tqdm(image_files, desc="Generating augmented dataset"):
        mask_path = train_masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        base_name = image_path.stem

        original_image_out = out_images_dir / f"{base_name}__orig.jpg"
        original_mask_out = out_masks_dir / f"{base_name}__orig.png"
        image.save(original_image_out, quality=95)
        save_mask(mask, original_mask_out)

        method_name, method_fn = select_augmentation_for_index(processed, augmentations)
        aug_image, aug_mask = method_fn(image, mask)
        aug_image_out = out_images_dir / f"{base_name}__{method_name}.jpg"
        aug_mask_out = out_masks_dir / f"{base_name}__{method_name}.png"

        aug_image.save(aug_image_out, quality=95)
        save_mask(aug_mask, aug_mask_out)

        debug_records.append(
            {
                "image_path": str(aug_image_out),
                "mask_path": str(aug_mask_out),
                "source": image_path.name,
                "method": method_name,
            }
        )
        method_counts[method_name] += 1

        processed += 1

    plot_path = make_debug_plot(
        records=debug_records,
        output_path=plot_output_path,
        num_samples=args.num_plot_samples,
    )

    total_output_images = len(list(out_images_dir.glob("*.jpg")))
    total_output_masks = len(list(out_masks_dir.glob("*.png")))

    print("\nAugmentation generation complete.")
    print(f"Processed pairs: {processed}")
    print(f"Skipped (missing mask): {skipped}")
    print(f"Output images: {total_output_images} -> {out_images_dir}")
    print(f"Output masks: {total_output_masks} -> {out_masks_dir}")
    if method_counts:
        print("Assigned augmentation counts:")
        for method_name, count in sorted(method_counts.items()):
            print(f"  {method_name}: {count}")
    if plot_path is not None:
        print(f"Debug plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
