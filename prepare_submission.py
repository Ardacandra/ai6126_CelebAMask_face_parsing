import argparse
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image


def get_project_root():
    return Path(__file__).resolve().parent


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_run_id(config):
    run_id = config.get("output", {}).get("run_id")
    if run_id:
        return str(run_id)
    return None


def read_latest_run_id(output_root):
    latest_path = os.path.join(output_root, "latest_run_id.txt")
    if not os.path.exists(latest_path):
        return None
    with open(latest_path, "r") as f:
        return f.read().strip() or None


def get_run_dir(output_root, run_id):
    return os.path.join(output_root, str(run_id))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare submission masks from generated predictions using config.yaml settings."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID override. If omitted, uses output.run_id or latest_run_id.txt.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Optional path to prediction masks. Defaults to <output.dir>/<run_id>/val_predictions.",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <output.dir>/<run_id>/submission.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create <run_id>_submission.zip.",
    )
    return parser.parse_args()


def prepare_masks(predictions_dir, masks_dir, output_size=(512, 512)):
    os.makedirs(masks_dir, exist_ok=True)

    copied_count = 0
    skipped = []

    for filename in sorted(os.listdir(predictions_dir)):
        src_path = os.path.join(predictions_dir, filename)
        if not os.path.isfile(src_path):
            continue

        if filename.endswith("_pred.png"):
            target_name = filename.replace("_pred.png", ".png")
        elif filename.endswith(".png"):
            target_name = filename
        else:
            skipped.append(filename)
            continue

        dst_path = os.path.join(masks_dir, target_name)

        with Image.open(src_path) as img:
            if img.size != output_size:
                img = img.resize(output_size, resample=Image.NEAREST)

            img.save(dst_path)
        copied_count += 1

    return copied_count, skipped


def main():
    args = parse_args()

    project_root = get_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    output_root = config["output"]["dir"]
    output_root_abs = os.path.join(project_root, output_root)

    run_id = args.run_id
    if not run_id:
        run_id = resolve_run_id(config)
    if not run_id:
        run_id = read_latest_run_id(output_root_abs)
    if not run_id:
        raise ValueError(
            "No run_id found. Set output.run_id in config.yaml, pass --run-id, or train first."
        )

    run_dir = get_run_dir(output_root_abs, run_id)

    if args.predictions_dir:
        predictions_dir = args.predictions_dir
        if not os.path.isabs(predictions_dir):
            predictions_dir = os.path.join(project_root, predictions_dir)
    else:
        predictions_dir = os.path.join(run_dir, "val_predictions")

    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(
            f"Predictions folder not found: {predictions_dir}. Run `python src/evaluate.py` first or pass --predictions-dir."
        )

    if args.submission_dir:
        submission_dir = args.submission_dir
        if not os.path.isabs(submission_dir):
            submission_dir = os.path.join(project_root, submission_dir)
    else:
        submission_dir = os.path.join(run_dir, "submission")

    masks_dir = os.path.join(submission_dir, "masks")

    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)

    copied_count, skipped = prepare_masks(predictions_dir, masks_dir)

    if copied_count == 0:
        raise ValueError(
            f"No PNG predictions found in {predictions_dir}. Expected files like *_pred.png."
        )

    print("\n" + "=" * 60)
    print("Submission Prepared")
    print("=" * 60)
    print(f"run_id         : {run_id}")
    print(f"predictions    : {predictions_dir}")
    print(f"submission_dir : {submission_dir}")
    print(f"masks_count    : {copied_count}")

    if skipped:
        print(f"skipped_files  : {len(skipped)}")

    if not args.no_zip:
        zip_base = os.path.join(run_dir, f"{run_id}_submission")
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=submission_dir)
        print(f"zip_file       : {zip_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
