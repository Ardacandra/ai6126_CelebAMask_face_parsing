import os
import warnings
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from helper import (
    CelebAMaskDataset,
    compute_class_weights,
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
)

warnings.filterwarnings("ignore")


def main():
    config = load_config()
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
            f.write("epoch,train_loss,val_loss\n")

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
    model_name = config.get("model", {}).get("name", "srresnet")
    max_params = config["model"].get("max_params", 1821085)

    log("\n" + "=" * 60)
    log("CelebAMask Face Parsing Training")
    log("=" * 60)
    log(f"Run ID: {run_id}")
    log(f"Output dir: {run_dir}")

    train_dataset = CelebAMaskDataset(
        config["data"]["train"]["images"],
        config["data"]["train"]["masks"],
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
        train_dataset, val_dataset = split_train_val(
            train_dataset, val_split=val_split, seed=split_seed
        )
        has_val_labels = True
        log("\nVal labels not found; using train/val split from training data.")
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

    # Compute class weights if enabled
    use_class_weights = config["training"].get("use_class_weights", False)
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset, num_classes=19, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        log("\n✓ Using weighted CrossEntropyLoss to handle class imbalance")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    checkpoint_every = config["training"].get("checkpoint_every", 1)
    patience = config["training"].get("patience", 10)

    log("\nTraining Configuration:")
    log(f"  Epochs: {num_epochs}")
    log(f"  Learning rate: {learning_rate}")
    log(f"  Device: {device}")
    log(f"  Checkpoint every: {checkpoint_every} epoch(s)")
    log(f"  Early stopping patience: {patience}")
    log("\n" + "=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        log(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if has_val_labels and val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            log(f"  Train Loss: {train_loss:.4f}")
            log(f"  Val Loss: {val_loss:.4f}")

            with open(metrics_file, "a") as f:
                f.write(f"{epoch + 1},{train_loss:.20f},{val_loss:.20f}\n")

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
                f.write(f"{epoch + 1},{train_loss:.20f},\n")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            log(f"  ✓ Checkpoint saved to {checkpoint_path}")

    log("\n" + "=" * 60)
    log("Training Complete!")
    log("Training ended")
    log("=" * 60)


if __name__ == "__main__":
    main()
