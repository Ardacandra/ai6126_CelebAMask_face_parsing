import os
import sys
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model import SRResNetFaceParsing, SRResNetFaceParsingV2

MODEL_REGISTRY = {
    "srresnet": SRResNetFaceParsing,
    "srresnet_v2": SRResNetFaceParsingV2,
}


def get_project_root():
    return project_root


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
    arch_cfg = model_cfg.get("arch_configs", {}).get(model_name, {})

    if model_name == "srresnet":
        num_residual_blocks = arch_cfg.get(
            "num_residual_blocks", model_cfg.get("num_residual_blocks", 8)
        )
        return SRResNetFaceParsing(
            num_classes=num_classes,
            num_residual_blocks=num_residual_blocks,
        )

    if model_name == "srresnet_v2":
        num_residual_blocks = arch_cfg.get(
            "num_residual_blocks", model_cfg.get("num_residual_blocks", 10)
        )
        channels = arch_cfg.get("channels", model_cfg.get("channels", 96))
        decoder_channels = arch_cfg.get(
            "decoder_channels", model_cfg.get("decoder_channels", [48, 24])
        )
        return SRResNetFaceParsingV2(
            num_classes=num_classes,
            num_residual_blocks=num_residual_blocks,
            channels=channels,
            decoder_channels=tuple(decoder_channels),
        )

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

        self.mask_transform = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)

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
            mask_pil = Image.open(mask_path)
            
            # Resize mask (keeps palette indices intact)
            mask_pil = self.mask_transform(mask_pil)
            
            # Extract raw palette indices as class labels
            mask = np.array(mask_pil, dtype=np.uint8)
            
            # Convert to tensor
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
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


def compute_class_weights(dataset, num_classes=19, device='cpu'):
    """Compute class weights for handling imbalanced data"""
    print("\nComputing class weights from dataset...")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    # Sample masks to compute class distribution
    sample_size = min(len(dataset), 500)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="Sampling class distribution"):
        _, mask, _ = dataset[idx]
        if mask is not None:
            mask_np = mask.numpy().astype(int).flatten()
            for class_id in range(num_classes):
                class_counts[class_id] += (mask_np == class_id).sum()
    
    # Compute inverse frequency weights
    total_pixels = class_counts.sum()
    class_weights = np.zeros(num_classes, dtype=np.float32)
    
    for i in range(num_classes):
        if class_counts[i] > 0:
            # Inverse frequency with smoothing
            class_weights[i] = total_pixels / (num_classes * class_counts[i])
        else:
            class_weights[i] = 0.0
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("Class weights computed:")
    for i in range(num_classes):
        if class_counts[i] > 0:
            pct = 100.0 * class_counts[i] / total_pixels
            print(f"  Class {i:2d}: weight={class_weights[i]:.3f} (freq={pct:.2f}%)")
    
    return torch.FloatTensor(class_weights).to(device)


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


def generate_predictions(model, dataloader, output_dir, device, output_size=(512, 512)):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Define CelebAMask-HQ color palette for 19 classes
    PALETTE = np.array([[i, i, i] for i in range(256)])
    PALETTE[:19] = np.array([
        [0, 0, 0],          # 0: background
        [204, 0, 0],        # 1: skin
        [76, 153, 0],       # 2: nose
        [204, 204, 0],      # 3: eye_g (glasses)
        [51, 51, 255],      # 4: l_eye
        [204, 0, 204],      # 5: r_eye
        [0, 255, 255],      # 6: l_brow
        [255, 204, 204],    # 7: r_brow
        [102, 51, 0],       # 8: l_ear
        [255, 0, 0],        # 9: r_ear
        [102, 204, 0],      # 10: mouth
        [255, 255, 0],      # 11: u_lip
        [0, 0, 153],        # 12: l_lip
        [0, 0, 204],        # 13: hair
        [255, 51, 153],     # 14: hat
        [0, 204, 204],      # 15: ear_ring
        [0, 51, 0],         # 16: neck_lace
        [255, 153, 51],     # 17: neck
        [0, 204, 0],        # 18: cloth
    ])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            images = batch[0].to(device)
            img_files = batch[2]

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            for i, img_file in enumerate(img_files):
                pred_mask = predictions[i].cpu().numpy().astype(np.uint8)
                output_path = os.path.join(output_dir, img_file.replace(".jpg", "_pred.png"))
                
                # Create palette-indexed image
                mask_img = Image.fromarray(pred_mask)
                if mask_img.size != output_size:
                    mask_img = mask_img.resize(output_size, resample=Image.NEAREST)
                mask_img.putpalette(PALETTE.reshape(-1).tolist())
                mask_img.save(output_path)


def split_train_val(dataset, val_split=0.2, seed=42):
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be between 0 and 1")

    num_samples = len(dataset)
    val_size = int(num_samples * val_split)
    if val_size < 1:
        val_size = 1
    if val_size >= num_samples:
        val_size = num_samples - 1

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
