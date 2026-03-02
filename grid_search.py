import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def parse_csv_list(value, cast_type):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("List argument cannot be empty")
    return [cast_type(item) for item in items]


def parse_scheduler(value):
    scheduler = value.strip().lower()
    if scheduler in {"none", "off", "disabled"}:
        return "none"
    return scheduler


def make_run_id(base_run_id, search_id, index):
    return f"{base_run_id}_{search_id}_{index:03d}"


def apply_combo_to_config(config, combo, run_id):
    cfg = copy.deepcopy(config)

    training_cfg = cfg.setdefault("training", {})
    output_cfg = cfg.setdefault("output", {})
    loss_cfg = training_cfg.setdefault("loss", {})
    ce_dice_boundary_cfg = loss_cfg.setdefault("ce_dice_boundary", {})
    scheduler_cfg = training_cfg.setdefault("scheduler", {})

    training_cfg["image_size"] = combo["image_size"]
    training_cfg["batch_size"] = combo["batch_size"]
    training_cfg["learning_rate"] = combo["learning_rate"]

    loss_cfg["name"] = "ce_dice_boundary"
    ce_dice_boundary_cfg["ce_weight"] = combo["ce_weight"]
    ce_dice_boundary_cfg["dice_weight"] = combo["dice_weight"]
    ce_dice_boundary_cfg["boundary_weight"] = combo["boundary_weight"]
    ce_dice_boundary_cfg["dilation"] = combo["dilation"]
    ce_dice_boundary_cfg["dice_smooth"] = combo["dice_smooth"]

    if combo["scheduler"] == "none":
        scheduler_cfg["enabled"] = False
    else:
        scheduler_cfg["enabled"] = True
        scheduler_cfg["type"] = combo["scheduler"]

    output_cfg["run_id"] = run_id

    return cfg


def read_best_metrics(metrics_file):
    if not metrics_file.exists():
        return None, None, None

    best_val_f1 = None
    best_val_loss = None
    best_epoch = None

    with metrics_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_value = row.get("epoch")
            val_loss_value = row.get("val_loss")
            val_f1_value = row.get("val_f1")

            try:
                epoch = int(epoch_value)
            except (TypeError, ValueError):
                continue

            val_loss = None
            val_f1 = None

            if val_loss_value not in {None, ""}:
                try:
                    val_loss = float(val_loss_value)
                except ValueError:
                    val_loss = None

            if val_f1_value not in {None, ""}:
                try:
                    val_f1 = float(val_f1_value)
                except ValueError:
                    val_f1 = None

            if val_f1 is not None and (best_val_f1 is None or val_f1 > best_val_f1):
                best_val_f1 = val_f1
                best_epoch = epoch

            if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss = val_loss

    return best_epoch, best_val_loss, best_val_f1


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run one-shot grid search over training hyperparameters"
    )
    parser.add_argument("--base-config", default="config.yaml", help="Base config YAML")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch training",
    )

    parser.add_argument("--image-sizes", default="512", help="Comma-separated image sizes")
    parser.add_argument("--batch-sizes", default="4", help="Comma-separated batch sizes")
    parser.add_argument(
        "--learning-rates", default="0.001", help="Comma-separated learning rates"
    )
    parser.add_argument(
        "--schedulers",
        default="reduce_on_plateau",
        help="Comma-separated scheduler types (use 'none' to disable)",
    )

    parser.add_argument("--ce-weights", default="1.0", help="Comma-separated CE weights")
    parser.add_argument("--dice-weights", default="1.0", help="Comma-separated Dice weights")
    parser.add_argument(
        "--boundary-weights", default="1.5", help="Comma-separated boundary weights"
    )
    parser.add_argument("--dilations", default="5", help="Comma-separated dilation values")
    parser.add_argument(
        "--dice-smooth-values", default="1.0", help="Comma-separated dice_smooth values"
    )

    parser.add_argument(
        "--search-id",
        default=None,
        help="Optional search id suffix for output folders",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip combinations whose run directory already exists",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base_config_path = (project_root / args.base_config).resolve()

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with base_config_path.open("r") as f:
        base_config = yaml.safe_load(f)

    image_sizes = parse_csv_list(args.image_sizes, int)
    batch_sizes = parse_csv_list(args.batch_sizes, int)
    learning_rates = parse_csv_list(args.learning_rates, float)
    schedulers = [parse_scheduler(item) for item in args.schedulers.split(",") if item.strip()]

    ce_weights = parse_csv_list(args.ce_weights, float)
    dice_weights = parse_csv_list(args.dice_weights, float)
    boundary_weights = parse_csv_list(args.boundary_weights, float)
    dilations = parse_csv_list(args.dilations, int)
    dice_smooth_values = parse_csv_list(args.dice_smooth_values, float)

    combos = list(
        itertools.product(
            image_sizes,
            batch_sizes,
            learning_rates,
            schedulers,
            ce_weights,
            dice_weights,
            boundary_weights,
            dilations,
            dice_smooth_values,
        )
    )

    if not combos:
        raise ValueError("No combinations generated for grid search")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_id = args.search_id.strip() if args.search_id else f"grid_{timestamp}"

    output_dir = Path(base_config.get("output", {}).get("dir", "out"))
    base_run_id = str(base_config.get("output", {}).get("run_id", "run"))

    search_root = output_dir / "grid_search" / search_id
    configs_dir = search_root / "configs"
    logs_dir = search_root / "logs"
    results_csv = search_root / "results.csv"
    results_json = search_root / "results.json"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid search id: {search_id}")
    print(f"Total combinations: {len(combos)}")
    print(f"Search root: {search_root}")

    rows = []
    for index, values in enumerate(combos, start=1):
        combo = {
            "image_size": values[0],
            "batch_size": values[1],
            "learning_rate": values[2],
            "scheduler": values[3],
            "ce_weight": values[4],
            "dice_weight": values[5],
            "boundary_weight": values[6],
            "dilation": values[7],
            "dice_smooth": values[8],
        }

        run_id = make_run_id(base_run_id, search_id, index)
        run_dir = output_dir / run_id

        if args.skip_completed and run_dir.exists():
            best_epoch, best_val_loss, best_val_f1 = read_best_metrics(run_dir / "metrics.csv")
            row = {
                "index": index,
                **combo,
                "status": "skipped",
                "return_code": 0,
                "elapsed_sec": 0.0,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "config_path": str(configs_dir / f"combo_{index:03d}.yaml"),
                "log_path": str(logs_dir / f"combo_{index:03d}.log"),
            }
            rows.append(row)
            print(f"[{index}/{len(combos)}] skipped existing run: {run_id}")
            continue

        run_config = apply_combo_to_config(base_config, combo, run_id)

        config_path = configs_dir / f"combo_{index:03d}.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(run_config, f, sort_keys=False)

        log_path = logs_dir / f"combo_{index:03d}.log"
        cmd = [
            args.python_executable,
            "src/train.py",
            "--config",
            str(config_path),
        ]

        print(
            f"[{index}/{len(combos)}] running run_id={run_id} "
            f"(img={combo['image_size']}, bs={combo['batch_size']}, lr={combo['learning_rate']}, "
            f"sched={combo['scheduler']})"
        )

        start_time = time.time()
        with log_path.open("w") as log_file:
            process = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        elapsed = time.time() - start_time

        metrics_file = run_dir / "metrics.csv"
        best_epoch, best_val_loss, best_val_f1 = read_best_metrics(metrics_file)

        status = "ok" if process.returncode == 0 else "failed"
        row = {
            "index": index,
            **combo,
            "status": status,
            "return_code": process.returncode,
            "elapsed_sec": round(elapsed, 2),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_f1": best_val_f1,
            "config_path": str(config_path),
            "log_path": str(log_path),
        }
        rows.append(row)

        with results_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

        with results_json.open("w") as f:
            json.dump(rows, f, indent=2)

        print(
            f"[{index}/{len(combos)}] {status} run_id={run_id} "
            f"elapsed={elapsed:.1f}s best_val_f1={best_val_f1}"
        )

    successful = [r for r in rows if r["status"] in {"ok", "skipped"} and r["best_val_f1"] is not None]
    if successful:
        best = max(successful, key=lambda r: r["best_val_f1"])
        print("\nBest configuration by val_f1:")
        print(json.dumps(best, indent=2))

    print(f"\nGrid search completed. Results saved to: {results_csv}")


if __name__ == "__main__":
    main()
