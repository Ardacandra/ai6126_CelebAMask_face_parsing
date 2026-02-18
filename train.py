import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============ Dataset ============
class CelebAMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, image_size=512, is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.is_train = is_train
        self.has_masks = masks_dir is not None and os.path.exists(masks_dir)
        
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        # Transform for images
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
        mask_file = img_file.replace('.jpg', '.png')
        
        img_path = os.path.join(self.images_dir, img_file)
        
        img = Image.open(img_path).convert('RGB')
        img = self.image_transform(img)
        
        if self.has_masks:
            mask_path = os.path.join(self.masks_dir, mask_file)
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
            return img, mask, img_file
        else:
            return img, None, img_file

# ============ SRResNet-based Face Parsing Model ============
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SRResNetFaceParsing(nn.Module):
    def __init__(self, num_classes=19, num_residual_blocks=8):
        super(SRResNetFaceParsing, self).__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction layer
        self.pred_head = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, x):
        x = self.conv_first(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        x = self.pred_head(x)
        return x

# ============ Count Parameters ============
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# ============ Training Functions ============
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks, _ in tqdm(dataloader, desc='Training'):
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
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch[0].to(device)
            masks = batch[1]
            
            if masks is None:
                # No masks available, skip loss calculation
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
        for batch in tqdm(dataloader, desc='Generating Predictions'):
            images = batch[0].to(device)
            img_files = batch[2]
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # Get class predictions
            
            for i, img_file in enumerate(img_files):
                pred_mask = predictions[i].cpu().numpy().astype(np.uint8)
                output_path = os.path.join(output_dir, img_file.replace('.jpg', '_pred.png'))
                Image.fromarray(pred_mask).save(output_path)

# ============ Main Training ============
def main():
    # Create output directory
    output_dir = config['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    image_size = config['training']['image_size']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    num_classes = config['model']['num_classes']
    max_params = config['model'].get('max_params', 1821085)
    
    print("\n" + "="*60)
    print("CelebAMask Face Parsing Training")
    print("="*60)
    
    # Load datasets
    train_dataset = CelebAMaskDataset(
        config['data']['train']['images'],
        config['data']['train']['masks'],
        image_size=image_size,
        is_train=True
    )
    
    val_masks_dir = config['data']['val'].get('masks', None)
    val_dataset = CelebAMaskDataset(
        config['data']['val']['images'],
        masks_dir=val_masks_dir,
        image_size=image_size,
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\nDataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Batch size: {batch_size}")
    
    # Create model
    model = SRResNetFaceParsing(num_classes=num_classes, num_residual_blocks=8)
    model = model.to(device)
    
    # Check parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Max allowed: {max_params:,}")
    
    if trainable_params > max_params:
        print(f"\nERROR: Model has {trainable_params:,} trainable parameters, exceeds limit of {max_params:,}")
        print("Please reduce model complexity.")
        return
    else:
        print(f"✓ Model parameters within limit ({trainable_params:,} < {max_params:,})")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print("\n" + "="*60)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Only validate if we have validation masks
        if val_dataset.has_masks:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"  ✓ Best model saved to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        else:
            print(f"  Train Loss: {train_loss:.4f}")
            # Save model every epoch if no validation set
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Model saved to {model_path}")
    
    # Load best model and generate predictions
    print("\n" + "="*60)
    print("Generating Predictions on Validation Set")
    print("="*60)
    
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    pred_output_dir = os.path.join(output_dir, 'val_predictions')
    generate_predictions(model, val_loader, pred_output_dir, device)
    
    print(f"\n✓ Predictions saved to {pred_output_dir}")
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
