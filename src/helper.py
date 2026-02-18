import os
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from model import SRResNetFaceParsing

MODEL_REGISTRY = {
    "srresnet": SRResNetFaceParsing,
}


def get_project_root():
    return Path(__file__).resolve().parents[1]


def load_config(config_path=None):
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(config):
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "srresnet")).lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    num_classes = model_cfg.get("num_classes", 19)

    if model_name == "srresnet":
        return SRResNetFaceParsing(num_classes=num_classes)

    raise ValueError(f"Unsupported model name: {model_name}")


def resolve_run_id(config, output_root, allow_generate=True):
    run_id = config.get("output", {}).get("run_id")
    if run_id:
        return str(run_id)
    if not allow_generate:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def get_run_dir(output_root, run_id):
    return os.path.join(output_root, str(run_id))


def write_latest_run_id(output_root, run_id):
    os.makedirs(output_root, exist_ok=True)
    latest_path = os.path.join(output_root, "latest_run_id.txt")
    with open(latest_path, "w") as f:
        f.write(str(run_id))


def read_latest_run_id(output_root):
    latest_path = os.path.join(output_root, "latest_run_id.txt")
    if not os.path.exists(latest_path):
        return None
    with open(latest_path, "r") as f:
        return f.read().strip() or None


class CelebAMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, image_size=512, is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.is_train = is_train
        self.has_masks = masks_dir is not None and os.path.exists(masks_dir)

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

        if is_train:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = img_file.replace(".jpg", ".png")

        img_path = os.path.join(self.images_dir, img_file)

        img = Image.open(img_path).convert("RGB")
        img = self.image_transform(img)

        if self.has_masks:
            mask_path = os.path.join(self.masks_dir, mask_file)
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            return img, mask, img_file
        else:
            return img, None, img_file



def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = [item[1] for item in batch]
    img_files = [item[2] for item in batch]

    if all(m is None for m in masks):
        masks = None
    else:
        masks = torch.stack([m for m in masks if m is not None])

    return images, masks, img_files


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, masks, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device).long().squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch[0].to(device)
            masks = batch[1]

            if masks is None:
                continue

            masks = masks.to(device).long().squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def generate_predictions(model, dataloader, output_dir, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            images = batch[0].to(device)
            img_files = batch[2]

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            for i, img_file in enumerate(img_files):
                pred_mask = predictions[i].cpu().numpy().astype(np.uint8)
                output_path = os.path.join(output_dir, img_file.replace(".jpg", "_pred.png"))
                Image.fromarray(pred_mask).save(output_path)
