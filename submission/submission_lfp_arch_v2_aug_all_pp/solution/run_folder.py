import argparse
import subprocess
import sys
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(input_dir: Path, recursive: bool):
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file()]

    images = [p for p in files if p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(images)


def main(input_dir, output_dir, weights, run_script, recursive=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    run_script = Path(run_script)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not run_script.exists() or not run_script.is_file():
        raise FileNotFoundError(f"run.py not found: {run_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir, recursive=recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    total = len(image_paths)
    print(f"Found {total} images.")

    for idx, image_path in enumerate(image_paths, start=1):
        mask_path = output_dir / f"{image_path.stem}.png"
        cmd = [
            sys.executable,
            str(run_script),
            "--input",
            str(image_path),
            "--output",
            str(mask_path),
            "--weights",
            str(weights),
        ]
        subprocess.run(cmd, check=True)
        print(f"[{idx}/{total}] Saved: {mask_path.name}")

    print(f"Done. Masks saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run solution/run.py for every image in an input folder"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    parser.add_argument("--run-script", type=str, default="run.py")
    parser.add_argument("--recursive", action="store_true")

    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights=args.weights,
        run_script=args.run_script,
        recursive=args.recursive,
    )
