import os
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from helper import (
    CelebAMaskDataset,
    create_model,
    custom_collate_fn,
    count_parameters,
    get_device,
    load_config,
    get_run_dir,
    read_latest_run_id,
    resolve_run_id,
    split_train_val,
    train_epoch,
    validate,
    write_latest_run_id,
    create_loss_fn,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train face parsing model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config.yaml at project root)",
    )
    return parser.parse_args()


def main(config_path=None):
    config = load_config(config_path=config_path)
    device = get_device()
    print(f"Using device: {device}")

    output_root = config["output"]["dir"]
    run_id = resolve_run_id(config, output_root, allow_generate=True)
    run_dir = get_run_dir(output_root, run_id)
    log_dir = os.path.join(run_dir, "logs")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    write_latest_run_id(output_root, run_id)

    log_file = os.path.join(log_dir, "train.log")
    metrics_file = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as f:
            f.write("epoch,train_loss,val_loss,val_f1\n")

    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_with_time = f"[{timestamp}] {message}"
        print(message_with_time)
        with open(log_file, "a") as f:
            f.write(message_with_time + "\n")

    log("Training started")

    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    model_name = config.get("model", {}).get("name", "srresnet_baseline")
    max_params = config["model"].get("max_params", 1821085)

    log("\n" + "=" * 60)
    log("CelebAMask Face Parsing Training")
    log("=" * 60)
    log(f"Run ID: {run_id}")
    log(f"Output dir: {run_dir}")

    train_images_dir = config["data"]["train"]["images"]
    train_masks_dir = config["data"]["train"]["masks"]
    original_train_images_dir = train_images_dir
    original_train_masks_dir = train_masks_dir
    aug_cfg = config.get("data", {}).get("augmentation", {})
    use_train_aug = aug_cfg.get("use_train_aug", False)
    if use_train_aug:
        train_aug_cfg = aug_cfg.get("train_aug", {})
        train_images_dir = train_aug_cfg.get("images", train_images_dir)
        train_masks_dir = train_aug_cfg.get("masks", train_masks_dir)

    train_dataset = CelebAMaskDataset(
        train_images_dir,
        train_masks_dir,
        image_size=image_size,
        is_train=True,
    )

    val_masks_dir = config["data"]["val"].get("masks", None)
    val_dataset = CelebAMaskDataset(
        config["data"]["val"]["images"],
        masks_dir=val_masks_dir,
        image_size=image_size,
        is_train=False,
    )

    has_val_labels = val_dataset.has_masks
    if not has_val_labels:
        val_split = config["training"].get("val_split", 0.2)
        split_seed = config["training"].get("split_seed", 42)
        if use_train_aug:
            original_dataset = CelebAMaskDataset(
                original_train_images_dir,
                original_train_masks_dir,
                image_size=image_size,
                is_train=False,
            )
            _, val_dataset = split_train_val(
                original_dataset, val_split=val_split, seed=split_seed
            )

            val_group_keys = {
                Path(original_dataset.image_files[idx]).stem.split("__", 1)[0]
                for idx in val_dataset.indices
            }
            train_indices = [
                idx
                for idx, image_file in enumerate(train_dataset.image_files)
                if Path(image_file).stem.split("__", 1)[0] not in val_group_keys
            ]
            if not train_indices:
                raise ValueError(
                    "No training samples left after excluding validation groups from augmented data"
                )
            train_dataset = Subset(train_dataset, train_indices)
            log("\nVal labels not found; using original-only val split with augmentation-aware train filtering.")
            log(f"  Original train samples for split: {len(original_dataset)}")
            log(f"  Validation groups held out: {len(val_group_keys)}")
            log(f"  Remaining train samples after filter: {len(train_dataset)}")
        else:
            train_dataset, val_dataset = split_train_val(
                train_dataset, val_split=val_split, seed=split_seed
            )
            log("\nVal labels not found; using train/val split from training data.")
        has_val_labels = True
        log(f"  Val split: {val_split:.2f}")
        log(f"  Split seed: {split_seed}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )

    log("\nDataset Info:")
    log(f"  Train samples: {len(train_dataset)}")
    log(f"  Val samples: {len(val_dataset) if val_dataset is not None else 0}")
    log(f"  Using augmented train data: {use_train_aug}")
    log(f"  Train images dir: {train_images_dir}")
    log(f"  Train masks dir: {train_masks_dir}")
    log(f"  Image size: {image_size}x{image_size}")
    log(f"  Batch size: {batch_size}")

    model = create_model(config)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    log("\nModel Parameters:")
    log(f"  Model name: {model_name}")
    log(f"  Total parameters: {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")
    log(f"  Max allowed: {max_params:,}")

    if trainable_params > max_params:
        log(
            f"\nERROR: Model has {trainable_params:,} trainable parameters, exceeds limit of {max_params:,}"
        )
        log("Please reduce model complexity.")
        return
    else:
        log(f"✓ Model parameters within limit ({trainable_params:,} < {max_params:,})")

    criterion, loss_name = create_loss_fn(config)
    criterion = criterion.to(device)
    ignore_index = config.get("training", {}).get("loss", {}).get("ignore_index", None)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    scheduler_cfg = config["training"].get("scheduler", {})
    scheduler_enabled = scheduler_cfg.get("enabled", True)
    if scheduler_enabled:
        scheduler_type = scheduler_cfg.get("type", "reduce_on_plateau").lower()
        if scheduler_type == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 5),
                min_lr=scheduler_cfg.get("min_lr", 1e-6),
            )
        else:
            log(
                f"\nWarning: Unsupported scheduler type '{scheduler_type}'. Continuing without scheduler."
            )

    checkpoint_every = config["training"].get("checkpoint_every", 1)
    patience = config["training"].get("patience", 10)

    log("\nTraining Configuration:")
    log(f"  Epochs: {num_epochs}")
    log(f"  Learning rate: {learning_rate}")
    log(f"  Loss: {loss_name}")
    log(f"  Device: {device}")
    if scheduler is None:
        log("  LR scheduler: disabled")
    else:
        log(
            "  LR scheduler: ReduceLROnPlateau "
            f"(mode={scheduler_cfg.get('mode', 'min')}, "
            f"factor={scheduler_cfg.get('factor', 0.5)}, "
            f"patience={scheduler_cfg.get('patience', 5)}, "
            f"min_lr={scheduler_cfg.get('min_lr', 1e-6)})"
        )
    log(f"  Checkpoint every: {checkpoint_every} epoch(s)")
    log(f"  Early stopping patience: {patience}")
    log("\n" + "=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        log(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if has_val_labels and val_loader is not None:
            val_loss, val_f1 = validate(
                model,
                val_loader,
                criterion,
                device,
                return_f1=True,
                ignore_index=ignore_index,
            )
            log(f"  Train Loss: {train_loss:.4f}")
            log(f"  Val Loss: {val_loss:.4f}")
            log(f"  Val F1 (macro): {val_f1:.4f}")

            with open(metrics_file, "a") as f:
                f.write(f"{epoch + 1},{train_loss:.20f},{val_loss:.20f},{val_f1:.20f}\n")

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(run_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                log(f"  ✓ Best model saved to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        else:
            log(f"  Train Loss: {train_loss:.4f}")
            model_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            log(f"  ✓ Model saved to {model_path}")

            with open(metrics_file, "a") as f:
                f.write(f"{epoch + 1},{train_loss:.20f},,\n")

            if scheduler is not None:
                scheduler.step(train_loss)

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            log(f"  ✓ Checkpoint saved to {checkpoint_path}")

    if has_val_labels and val_loader is not None:
        model_path = os.path.join(run_dir, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            final_val_loss, final_val_f1 = validate(
                model,
                val_loader,
                criterion,
                device,
                return_f1=True,
                ignore_index=ignore_index,
            )
            log("\nFinal Best Model Validation Metrics:")
            log(f"  Val Loss: {final_val_loss:.4f}")
            log(f"  Val F1 (macro): {final_val_f1:.4f}")

    log("\n" + "=" * 60)
    log("Training Complete!")
    log("Training ended")
    log("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)
