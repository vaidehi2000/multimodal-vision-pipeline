import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import random
from pathlib import Path
import cv2
from torchvision import transforms

def custom_train_test_split(indices, test_size=0.2, val_size=0.1, random_seed=42):
    """Custom implementation of train/test/val split without scikit-learn dependency"""
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle indices
    indices = list(indices)
    random.shuffle(indices)
    
    # Calculate split sizes
    n_samples = len(indices)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices

def normalize_channel(channel):
    """Normalize a channel to 0-255 range"""
    min_val = np.min(channel)
    max_val = np.max(channel)
    if max_val == min_val:
        return np.zeros_like(channel, dtype=np.uint8)
    return ((channel - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes
    Args:
        batch: List of samples from dataset
    Returns:
        Dictionary with batched tensors
    """
    # Find max number of boxes in this batch
    max_boxes = max(len(sample['boxes']) for sample in batch)
    
    # Initialize batched tensors
    batched = {
        'rgb': [],
        'thermal': [],
        'event': [],
        'boxes': [],
        'image_path': [],
        'num_boxes': []  # Keep track of actual number of boxes
    }
    
    # Process each sample
    for sample in batch:
        batched['rgb'].append(sample['rgb'])
        batched['thermal'].append(sample['thermal'])
        batched['event'].append(sample['event'])
        batched['image_path'].append(sample['image_path'])
        
        # Get boxes and pad if necessary
        boxes = sample['boxes']
        num_boxes = len(boxes)
        batched['num_boxes'].append(num_boxes)
        
        if num_boxes < max_boxes:
            # Pad with zeros
            padding = torch.zeros((max_boxes - num_boxes, 5), dtype=boxes.dtype)
            boxes = torch.cat([boxes, padding], dim=0)
        
        batched['boxes'].append(boxes)
    
    # Stack tensors
    batched['rgb'] = torch.stack(batched['rgb'])
    batched['thermal'] = torch.stack(batched['thermal'])
    batched['event'] = torch.stack(batched['event'])
    batched['boxes'] = torch.stack(batched['boxes'])
    batched['num_boxes'] = torch.tensor(batched['num_boxes'])
    
    return batched

class TriAirDataset(Dataset):
    def __init__(self, data_root, transform=None, split='train', val_split=0.1, test_split=0.2, random_seed=42):
        """
        Args:
            data_root (str): Root directory of the dataset
            transform (callable, optional): Optional transform to be applied on RGB data
            split (str): One of 'train', 'val', or 'test'
            val_split (float): Validation split ratio
            test_split (float): Test split ratio
            random_seed (int): Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / 'images'
        self.labels_dir = self.data_root / 'labels'
        self.transform = transform
        self.split = split
        
        # Create transforms for different modalities
        self.rgb_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if transform else None
        
        # Simple normalization for thermal and event data
        self.thermal_transform = transforms.Normalize(mean=[0.5], std=[0.5]) if transform else None
        self.event_transform = transforms.Normalize(mean=[0.5], std=[0.5]) if transform else None
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.npy')])
        
        # Create indices for splitting
        indices = list(range(len(self.image_files)))
        train_indices, val_indices, test_indices = custom_train_test_split(
            indices, test_size=test_split, val_size=val_split, random_seed=random_seed
        )
        
        # Select appropriate split
        if split == 'train':
            self.image_files = [self.image_files[i] for i in train_indices]
        elif split == 'val':
            self.image_files = [self.image_files[i] for i in val_indices]
        elif split == 'test':
            self.image_files = [self.image_files[i] for i in test_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image data
        img_path = self.images_dir / self.image_files[idx]
        img_data = np.load(img_path)
        
        # Extract different modalities
        rgb = img_data[..., :3]  # BGR format
        thermal = img_data[..., 3]
        event = img_data[..., 4]
        
        # Load labels
        label_path = self.labels_dir / f"{self.image_files[idx].replace('.npy', '.txt')}"
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([class_id, x_center, y_center, width, height])
        
        boxes = np.array(boxes) if boxes else np.zeros((0, 5))
        
        # Convert to torch tensors and normalize RGB to [0, 1]
        rgb = torch.from_numpy(rgb).float() / 255.0  # Convert to float and normalize to [0, 1]
        rgb = rgb.permute(2, 0, 1)  # HWC to CHW
        
        # Convert thermal and event to float tensors and normalize to [0, 1]
        thermal = torch.from_numpy(thermal).float()
        event = torch.from_numpy(event).float()
        
        # Normalize thermal and event to [0, 1] range
        thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
        event = (event - event.min()) / (event.max() - event.min() + 1e-8)
        
        thermal = thermal.unsqueeze(0)  # Add channel dimension
        event = event.unsqueeze(0)  # Add channel dimension
        
        boxes = torch.from_numpy(boxes).float()
        
        # Apply transforms
        if self.transform:
            rgb = self.rgb_transform(rgb)
            thermal = self.thermal_transform(thermal)
            event = self.event_transform(event)
        
        return {
            'rgb': rgb,
            'thermal': thermal,
            'event': event,
            'boxes': boxes,
            'image_path': str(img_path)
        }
    
    def visualize_sample(self, idx, save_path=None):
        """
        Visualize a sample with bounding boxes drawn on all modalities
        Args:
            idx (int): Index of the sample to visualize
            save_path (str, optional): Path to save the visualization
        """
        # Load raw data for visualization
        img_path = self.images_dir / self.image_files[idx]
        frame_data = np.load(img_path)
        
        # Extract channels
        rgb = frame_data[..., :3]
        thermal = frame_data[..., 3]
        events = frame_data[..., 4]
        
        # Normalize thermal to 0-255
        thermal_norm = normalize_channel(thermal)
        
        # Create thermal heatmap
        thermal_colored = cv2.applyColorMap(thermal_norm, cv2.COLORMAP_JET)
        
        # Create event visualization
        event_vis = np.dstack([events, events, events])
        
        # Get dimensions
        h, w = rgb.shape[:2]
        
        # Create combined image with full frames side by side
        combined = np.hstack([rgb, thermal_colored, event_vis])
        
        # Load and draw bounding boxes
        label_path = self.labels_dir / f"{self.image_files[idx].replace('.npy', '.txt')}"
        if os.path.exists(label_path):
            bbox_data = np.loadtxt(label_path)
            if len(bbox_data.shape) == 1:
                bbox_data = bbox_data.reshape(1, -1)
                
            for bbox in bbox_data:
                class_id, x_center, y_center, width, height = bbox
                
                # Convert from normalized coordinates to pixel coordinates
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h
                
                # Draw on all modalities
                # RGB
                x1 = int(x_center_px - width_px/2)
                x2 = int(x_center_px + width_px/2)
                y1 = int(y_center_px - height_px/2)
                y2 = int(y_center_px + height_px/2)
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Thermal
                x1 = int(x_center_px - width_px/2) + w
                x2 = int(x_center_px + width_px/2) + w
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Events
                x1 = int(x_center_px - width_px/2) + 2*w
                x2 = int(x_center_px + width_px/2) + 2*w
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, combined)
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TriAir Dataset Visualization')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample to visualize')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the visualization (optional)')
    args = parser.parse_args()

    # Create dataset
    dataset = TriAirDataset(data_root=args.data_root)
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)} samples")
    
    # Handle save path
    save_path = args.save_path
    if save_path:
        if os.path.isdir(save_path):
            # If save_path is a directory, create a filename with the sample index
            save_path = os.path.join(save_path, f'sample_{args.sample_idx}.png')
        elif not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # If save_path doesn't have an image extension, add .png
            save_path = f"{save_path}.png"
    
    # Visualize sample
    print(f"Visualizing sample {args.sample_idx}")
    if save_path:
        print(f"Saving visualization to: {save_path}")
    dataset.visualize_sample(args.sample_idx, save_path=save_path)

if __name__ == '__main__':
    main() 