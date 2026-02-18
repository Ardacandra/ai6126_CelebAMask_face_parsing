import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up directories from config
train_images_dir = config['data']['train']['images']
train_masks_dir = config['data']['train']['masks']
output_dir = config['output']['dir']

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of image files
image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])

# Select number of samples to visualize
num_samples = config['visualization']['num_samples']
selected_samples = image_files[:num_samples]

# Create figure with subplots
figsize_width = config['visualization']['figsize_width']
figsize_height = num_samples * config['visualization']['figsize_height_per_sample']
fig, axes = plt.subplots(num_samples, 2, figsize=(figsize_width, figsize_height))

for idx, img_file in enumerate(selected_samples):
    # Get corresponding mask file
    mask_file = img_file.replace('.jpg', '.png')
    
    # Load image and mask
    img_path = os.path.join(train_images_dir, img_file)
    mask_path = os.path.join(train_masks_dir, mask_file)
    
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    
    # Plot image
    axes[idx, 0].imshow(img)
    axes[idx, 0].axis('off')
    if idx == 0:
        axes[idx, 0].set_title('Image', fontsize=14, fontweight='bold')
    
    # Plot mask
    axes[idx, 1].imshow(mask)
    axes[idx, 1].axis('off')
    if idx == 0:
        axes[idx, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(output_dir, 'train_samples_visualization.png')
plt.savefig(output_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
print(f"Visualization saved to: {output_path}")
plt.close()

print(f"\nVisualized {num_samples} training samples successfully!")
