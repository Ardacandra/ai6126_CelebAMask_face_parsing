import os
import warnings
import torch
from torch.utils.data import DataLoader

from helper import (
    CelebAMaskDataset,
    create_model,
    custom_collate_fn,
    generate_predictions,
    get_device,
    get_run_dir,
    load_config,
    read_latest_run_id,
    resolve_run_id,
)

warnings.filterwarnings("ignore")


def main():
    config = load_config()
    device = get_device()
    print(f"Using device: {device}")

    output_root = config["output"]["dir"]
    os.makedirs(output_root, exist_ok=True)
    run_id = resolve_run_id(config, output_root, allow_generate=False)
    if not run_id:
        run_id = read_latest_run_id(output_root)
    if not run_id:
        raise ValueError("No run_id found. Set output.run_id in config.yaml or train first.")

    run_dir = get_run_dir(output_root, run_id)

    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]

    val_masks_dir = config["data"]["val"].get("masks", None)
    val_dataset = CelebAMaskDataset(
        config["data"]["val"]["images"],
        masks_dir=val_masks_dir,
        image_size=image_size,
        is_train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    model = create_model(config)
    model = model.to(device)

    model_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Run training first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))

    print("\n" + "=" * 60)
    print("Generating Predictions on Validation Set")
    print("=" * 60)

    pred_output_dir = os.path.join(run_dir, "val_predictions")
    generate_predictions(model, val_loader, pred_output_dir, device)

    print(f"\n✓ Predictions saved to {pred_output_dir}")
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
