import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


PALETTE = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
PALETTE[:19] = np.array(
    [
        [0, 0, 0],
        [204, 0, 0],
        [76, 153, 0],
        [204, 204, 0],
        [51, 51, 255],
        [204, 0, 204],
        [0, 255, 255],
        [255, 204, 204],
        [102, 51, 0],
        [255, 0, 0],
        [102, 204, 0],
        [255, 255, 0],
        [0, 0, 153],
        [0, 0, 204],
        [255, 51, 153],
        [0, 204, 204],
        [0, 51, 0],
        [255, 153, 51],
        [0, 204, 0],
    ],
    dtype=np.uint8,
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_run_dir(config: dict, override_output_folder: str | None = None) -> Path:
    project_root = get_project_root()

    if override_output_folder:
        output_folder = Path(override_output_folder)
        if not output_folder.is_absolute():
            output_folder = project_root / output_folder
        return output_folder

    output_root = config["output"]["dir"]
    run_id = str(config["output"]["run_id"])
    return project_root / output_root / run_id


def read_index_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        return np.array(img, dtype=np.uint8)


def write_index_mask(mask: np.ndarray, out_path: Path):
    out_img = Image.fromarray(mask.astype(np.uint8), mode="P")
    out_img.putpalette(PALETTE.reshape(-1).tolist())
    out_img.save(out_path)


def small_component_removal(
    mask: np.ndarray,
    num_classes: int,
    min_size_default: int,
    min_size_by_class: dict,
) -> np.ndarray:
    result = mask.copy()

    # Operate class-by-class so tiny isolated islands are reassigned to local context.
    for class_id in range(1, num_classes):
        class_mask = (result == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        threshold = int(min_size_by_class.get(str(class_id), min_size_default))

        for component_id in range(1, n_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= threshold:
                continue

            component_mask = labels == component_id
            ys, xs = np.where(component_mask)
            if ys.size == 0:
                continue

            y0 = max(0, ys.min() - 1)
            y1 = min(result.shape[0], ys.max() + 2)
            x0 = max(0, xs.min() - 1)
            x1 = min(result.shape[1], xs.max() + 2)

            neighborhood = result[y0:y1, x0:x1]
            neighborhood_component = component_mask[y0:y1, x0:x1]
            neighbors = neighborhood[~neighborhood_component]

            if neighbors.size == 0:
                replacement = 0
            else:
                replacement = int(np.bincount(neighbors).argmax())

            result[component_mask] = replacement

    return result


def majority_filter_3x3(mask: np.ndarray, num_classes: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.float32)
    scores = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)

    for class_id in range(num_classes):
        class_binary = (mask == class_id).astype(np.float32)
        scores[class_id] = cv2.filter2D(class_binary, -1, kernel, borderType=cv2.BORDER_REFLECT)

    return np.argmax(scores, axis=0).astype(np.uint8)


def dense_crf_refinement(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    cfg: dict,
) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError as exc:
        raise RuntimeError(
            "DenseCRF requested but pydensecrf is not installed. "
            "Install with: pip install pydensecrf"
        ) from exc

    h, w = mask.shape
    confidence = float(cfg.get("unary_confidence", 0.9))
    gt_prob = np.clip(confidence, 1e-4, 1.0 - 1e-4)
    other_prob = (1.0 - gt_prob) / max(num_classes - 1, 1)

    probs = np.full((num_classes, h, w), other_prob, dtype=np.float32)
    for class_id in range(num_classes):
        probs[class_id][mask == class_id] = gt_prob

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    image_rgb = np.ascontiguousarray(image_rgb)

    crf = dcrf.DenseCRF2D(w, h, num_classes)
    crf.setUnaryEnergy(unary)

    gaussian_sxy = int(cfg.get("gaussian_sxy", 3))
    gaussian_compat = int(cfg.get("gaussian_compat", 3))
    bilateral_sxy = int(cfg.get("bilateral_sxy", 50))
    bilateral_srgb = int(cfg.get("bilateral_srgb", 13))
    bilateral_compat = int(cfg.get("bilateral_compat", 10))
    iterations = int(cfg.get("iterations", 5))

    crf.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)
    crf.addPairwiseBilateral(
        sxy=bilateral_sxy,
        srgb=bilateral_srgb,
        rgbim=image_rgb,
        compat=bilateral_compat,
    )

    q = np.array(crf.inference(iterations), dtype=np.float32)
    refined = np.argmax(q, axis=0).reshape(h, w).astype(np.uint8)
    return refined


def apply_method(
    method_name: str,
    mask: np.ndarray,
    image_rgb: np.ndarray,
    num_classes: int,
    cfg: dict,
) -> np.ndarray:
    post_cfg = cfg.get("postprocessing", {})
    scr_cfg = post_cfg.get("small_component_removal", {})
    dcrf_cfg = post_cfg.get("dense_crf", {})

    min_size_default = int(scr_cfg.get("min_size_default", 32))
    min_size_by_class = scr_cfg.get("min_size_by_class", {})

    if method_name == "small_component_removal":
        return small_component_removal(mask, num_classes, min_size_default, min_size_by_class)

    if method_name == "majority_filter_3x3":
        return majority_filter_3x3(mask, num_classes)

    if method_name == "dense_crf":
        return dense_crf_refinement(image_rgb, mask, num_classes, dcrf_cfg)

    if method_name == "combined_all":
        out = small_component_removal(mask, num_classes, min_size_default, min_size_by_class)
        out = majority_filter_3x3(out, num_classes)
        if bool(dcrf_cfg.get("enabled", False)):
            out = dense_crf_refinement(image_rgb, out, num_classes, dcrf_cfg)
        return out

    raise ValueError(
        "Unsupported postprocessing.method: "
        f"{method_name}. Supported: small_component_removal, majority_filter_3x3, dense_crf, combined_all"
    )


def build_submission(
    run_dir: Path,
    method_name: str,
    cfg: dict,
    config_path: Path,
):
    num_classes = int(cfg.get("model", {}).get("num_classes", 19))

    pred_dir = run_dir / "val_predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    out_dir = run_dir / f"submission_{method_name}" / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_images_dir = cfg.get("data", {}).get("val", {}).get("images", None)
    if not val_images_dir:
        raise ValueError("Missing config key data.val.images required for DenseCRF input images")
    val_images_dir = get_project_root() / val_images_dir

    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if not pred_files:
        raise ValueError(f"No PNG files found in {pred_dir}")

    dense_crf_needed = method_name == "dense_crf" or (
        method_name == "combined_all"
        and bool(cfg.get("postprocessing", {}).get("dense_crf", {}).get("enabled", False))
    )

    for pred_path in tqdm(pred_files, desc=f"Postprocessing ({method_name})"):
        in_mask = read_index_mask(pred_path)

        base_name = pred_path.name.replace("_pred.png", ".jpg")
        image_path = val_images_dir / base_name
        if not image_path.exists():
            alt_base_name = pred_path.stem + ".jpg"
            image_path = val_images_dir / alt_base_name

        if dense_crf_needed and not image_path.exists():
            raise FileNotFoundError(
                f"Could not map prediction {pred_path.name} to validation image in {val_images_dir}"
            )

        if image_path.exists():
            with Image.open(image_path) as img:
                image_rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        else:
            image_rgb = np.zeros((in_mask.shape[0], in_mask.shape[1], 3), dtype=np.uint8)

        out_mask = apply_method(method_name, in_mask, image_rgb, num_classes, cfg)
        out_name = pred_path.name.replace("_pred.png", ".png")
        write_index_mask(out_mask, out_dir / out_name)

    folder_name = run_dir.name
    zip_filename = f"{folder_name}_post_{method_name}_submission"
    zip_base = run_dir / zip_filename
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=out_dir.parent)

    print("\n" + "=" * 60)
    print("Postprocessing Complete")
    print("=" * 60)
    print(f"config          : {config_path}")
    print(f"run_dir         : {run_dir}")
    print(f"method          : {method_name}")
    print(f"input_dir       : {pred_dir}")
    print(f"output_dir      : {out_dir.parent}")
    print(f"output_masks    : {len(pred_files)}")
    print(f"zip_file        : {zip_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply postprocessing to out/<run_id>/val_predictions and write submission_<method>."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml at project root)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Optional override for run output folder (example: out/lfp_v4)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Optional override for postprocessing method",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = get_project_root()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)

    method_name = args.method or config.get("postprocessing", {}).get("method", "small_component_removal")
    run_dir = resolve_run_dir(config, override_output_folder=args.output_folder)

    build_submission(run_dir, method_name, config, config_path)


if __name__ == "__main__":
    main()
