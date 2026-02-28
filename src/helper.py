import os
import sys
import random
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model import LiteFaceParser, SRResNetBaseline

MODEL_REGISTRY = {
    "srresnet_baseline": SRResNetBaseline,
    "lite_face_parser": LiteFaceParser,
}


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets = targets.long()

        clamped_targets = torch.clamp(targets, min=0, max=num_classes - 1)
        one_hot = F.one_hot(clamped_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid_mask
            one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=None, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        targets = targets.long()
        ignore_index = self.ignore_index if self.ignore_index is not None else -100
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.class_weights,
            ignore_index=ignore_index,
        )
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss[valid_mask]

        if loss.numel() == 0:
            return ce_loss.mean() * 0.0
        return loss.mean()


class BoundaryAwareLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        boundary_weight=1.0,
        dilation=3,
        ignore_index=None,
        class_weights=None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.dilation = max(1, int(dilation))
        self.ignore_index = ignore_index

        ce_ignore_index = self.ignore_index if self.ignore_index is not None else -100
        ce_class_weights = None
        if class_weights is not None:
            ce_class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights, ignore_index=ce_ignore_index)

    def _target_boundaries(self, targets):
        edge_x = (targets[:, :, 1:] != targets[:, :, :-1]).float()
        edge_x = F.pad(edge_x, (0, 1, 0, 0))
        edge_y = (targets[:, 1:, :] != targets[:, :-1, :]).float()
        edge_y = F.pad(edge_y, (0, 0, 0, 1))
        boundaries = torch.clamp(edge_x + edge_y, 0.0, 1.0)

        if self.dilation > 1:
            boundaries = F.max_pool2d(
                boundaries.unsqueeze(1),
                kernel_size=self.dilation,
                stride=1,
                padding=self.dilation // 2,
            ).squeeze(1)
        return boundaries

    def _predicted_boundaries(self, probs):
        grad_x = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        edge_strength = (grad_x.mean(dim=1) + grad_y.mean(dim=1)).clamp(0.0, 1.0)
        return edge_strength

    def forward(self, logits, targets):
        targets = targets.long()
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        pred_edges = self._predicted_boundaries(probs)
        target_edges = self._target_boundaries(targets)

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).float()
            pred_edges = pred_edges * valid_mask
            target_edges = target_edges * valid_mask
            valid_count = valid_mask.sum().clamp_min(1.0)
            boundary_loss = F.binary_cross_entropy(pred_edges, target_edges, reduction="sum") / valid_count
        else:
            boundary_loss = F.binary_cross_entropy(pred_edges, target_edges)

        return self.ce_weight * ce_loss + self.boundary_weight * boundary_loss


class CEDiceBoundaryLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=1.0,
        dice_smooth=1.0,
        dilation=3,
        ignore_index=None,
        class_weights=None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

        ce_ignore_index = self.ignore_index if self.ignore_index is not None else -100
        ce_weights = None
        if class_weights is not None:
            ce_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=ce_ignore_index)
        self.dice = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
        self.boundary_helper = BoundaryAwareLoss(
            ce_weight=0.0,
            boundary_weight=1.0,
            dilation=dilation,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

    def forward(self, logits, targets):
        targets = targets.long()

        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        probs = torch.softmax(logits, dim=1)
        pred_edges = self.boundary_helper._predicted_boundaries(probs)
        target_edges = self.boundary_helper._target_boundaries(targets)

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).float()
            pred_edges = pred_edges * valid_mask
            target_edges = target_edges * valid_mask
            valid_count = valid_mask.sum().clamp_min(1.0)
            boundary_loss = F.binary_cross_entropy(pred_edges, target_edges, reduction="sum") / valid_count
        else:
            boundary_loss = F.binary_cross_entropy(pred_edges, target_edges)

        return (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )


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
    model_name = str(model_cfg.get("name", "srresnet_baseline")).lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    num_classes = model_cfg.get("num_classes", 19)
    arch_cfg = model_cfg.get("arch_configs", {}).get(model_name, {})

    if model_name == "srresnet_baseline":
        num_residual_blocks = arch_cfg.get(
            "num_residual_blocks", model_cfg.get("num_residual_blocks", 16)
        )
        context_dilations = arch_cfg.get(
            "context_dilations", model_cfg.get("context_dilations", [2, 4, 8])
        )
        decoder_channels = arch_cfg.get(
            "decoder_channels", model_cfg.get("decoder_channels", [64, 48, 32])
        )
        return SRResNetBaseline(
            num_classes=num_classes,
            num_residual_blocks=num_residual_blocks,
            context_dilations=tuple(context_dilations),
            decoder_channels=tuple(decoder_channels),
        )

    if model_name == "lite_face_parser":
        stage_channels = arch_cfg.get(
            "stage_channels", model_cfg.get("stage_channels", [32, 48, 64, 96])
        )
        expand_ratio = arch_cfg.get(
            "expand_ratio", model_cfg.get("expand_ratio", 4)
        )
        aspp_dilations = arch_cfg.get(
            "aspp_dilations", model_cfg.get("aspp_dilations", [1, 2, 4])
        )
        aspp_channels = arch_cfg.get(
            "aspp_channels", model_cfg.get("aspp_channels", 64)
        )
        return LiteFaceParser(
            num_classes=num_classes,
            stage_channels=tuple(stage_channels),
            expand_ratio=expand_ratio,
            aspp_dilations=tuple(aspp_dilations),
            aspp_channels=aspp_channels,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def create_loss_fn(config):
    loss_cfg = config.get("training", {}).get("loss", {})
    loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()
    ignore_index = loss_cfg.get("ignore_index", None)
    class_weights = loss_cfg.get("class_weights", None)

    ce_aliases = {"cross_entropy", "cross-entropy", "ce"}
    dice_aliases = {"dice", "dice_loss"}
    focal_aliases = {"focal", "focal_loss"}
    boundary_aliases = {"boundary", "boundary_aware", "boundary_aware_loss", "boundary-aware"}
    ce_dice_boundary_aliases = {
        "ce_dice_boundary",
        "cross_entropy_dice_boundary",
        "cross_entropy+dice+boundary_aware",
        "combo",
    }

    if loss_name in ce_aliases:
        ce_ignore_index = ignore_index if ignore_index is not None else -100
        ce_weights = None
        if class_weights is not None:
            ce_weights = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=ce_weights, ignore_index=ce_ignore_index), "cross_entropy"

    if loss_name in dice_aliases:
        dice_cfg = loss_cfg.get("dice", {})
        return (
            DiceLoss(
                smooth=float(dice_cfg.get("smooth", 1.0)),
                ignore_index=ignore_index,
            ),
            "dice",
        )

    if loss_name in focal_aliases:
        focal_cfg = loss_cfg.get("focal", {})
        return (
            FocalLoss(
                alpha=float(focal_cfg.get("alpha", 1.0)),
                gamma=float(focal_cfg.get("gamma", 2.0)),
                ignore_index=ignore_index,
                class_weights=class_weights,
            ),
            "focal",
        )

    if loss_name in boundary_aliases:
        boundary_cfg = loss_cfg.get("boundary_aware", {})
        return (
            BoundaryAwareLoss(
                ce_weight=float(boundary_cfg.get("ce_weight", 1.0)),
                boundary_weight=float(boundary_cfg.get("boundary_weight", 1.0)),
                dilation=int(boundary_cfg.get("dilation", 3)),
                ignore_index=ignore_index,
                class_weights=class_weights,
            ),
            "boundary_aware",
        )

    if loss_name in ce_dice_boundary_aliases:
        combo_cfg = loss_cfg.get("ce_dice_boundary", {})
        return (
            CEDiceBoundaryLoss(
                ce_weight=float(combo_cfg.get("ce_weight", 1.0)),
                dice_weight=float(combo_cfg.get("dice_weight", 1.0)),
                boundary_weight=float(combo_cfg.get("boundary_weight", 1.0)),
                dice_smooth=float(combo_cfg.get("dice_smooth", 1.0)),
                dilation=int(combo_cfg.get("dilation", 5)),
                ignore_index=ignore_index,
                class_weights=class_weights,
            ),
            "ce_dice_boundary",
        )

    raise ValueError(
        "Unsupported training.loss.name: "
        f"'{loss_name}'. Supported: cross_entropy, dice, focal, boundary_aware, ce_dice_boundary"
    )


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
    def __init__(
        self,
        images_dir,
        masks_dir=None,
        image_size=512,
        is_train=True,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.is_train = is_train
        self.has_masks = masks_dir is not None and os.path.exists(masks_dir)

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

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
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to split into train/val")

    image_files = getattr(dataset, "image_files", None)
    if image_files is None or len(image_files) != num_samples:
        raise ValueError("Dataset must expose image_files for group-aware splitting")

    target_val_size = int(num_samples * val_split)
    if target_val_size < 1:
        target_val_size = 1
    if target_val_size >= num_samples:
        target_val_size = num_samples - 1

    grouped_indices = {}
    for index, image_file in enumerate(image_files):
        stem = Path(image_file).stem
        group_key = stem.split("__", 1)[0]
        grouped_indices.setdefault(group_key, []).append(index)

    group_keys = list(grouped_indices.keys())
    random.Random(seed).shuffle(group_keys)

    val_indices = []
    for group_key in group_keys:
        candidate = grouped_indices[group_key]
        if len(val_indices) >= target_val_size:
            break
        if len(val_indices) + len(candidate) >= num_samples:
            continue
        val_indices.extend(candidate)

    if not val_indices:
        first_group = grouped_indices[group_keys[0]]
        if len(first_group) >= num_samples:
            raise ValueError("Unable to create leakage-free split: all samples belong to one group")
        val_indices = list(first_group)

    val_indices_set = set(val_indices)
    train_indices = [idx for idx in range(num_samples) if idx not in val_indices_set]

    if not train_indices:
        raise ValueError("Unable to create leakage-free split with non-empty train set")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
