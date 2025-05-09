import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import numpy as np
from dataset import TriAirDataset, collate_fn
from model import create_model
import torch.nn.functional as F

def generate_targets(boxes, feature_size, img_size):
    """
    Convert ground truth boxes to target format for the detection head
    Args:
        boxes: Tensor of shape [B, N, 5] containing [class_id, x_center, y_center, width, height]
        feature_size: Tuple of (H, W) for the feature map size
        img_size: Tuple of (H, W) for the original image size
    Returns:
        cls_target: Tensor of shape [B, H*W] containing classification targets
        reg_target: Tensor of shape [B, 4, H*W] containing regression targets
    """
    batch_size = boxes.shape[0]
    H, W = feature_size
    img_h, img_w = img_size
    
    # Initialize targets
    cls_target = torch.zeros((batch_size, H * W), device=boxes.device)
    reg_target = torch.zeros((batch_size, 4, H * W), device=boxes.device)
    
    # For each image in batch
    for b in range(batch_size):
        # Get boxes for this image
        img_boxes = boxes[b]
        
        # Skip if no boxes
        if len(img_boxes) == 0:
            continue
        
        # For each ground truth box
        for box in img_boxes:
            class_id, x_center, y_center, width, height = box
            
            # Convert normalized coordinates to grid coordinates
            grid_x = int(x_center * W)
            grid_y = int(y_center * H)
            
            # Ensure coordinates are within bounds
            grid_x = min(max(0, grid_x), W-1)
            grid_y = min(max(0, grid_y), H-1)
            
            # Set classification target
            idx = grid_y * W + grid_x
            cls_target[b, idx] = 1.0
            
            # Set regression target (in normalized coordinates)
            reg_target[b, 0, idx] = x_center  # x
            reg_target[b, 1, idx] = y_center  # y
            reg_target[b, 2, idx] = width     # w
            reg_target[b, 3, idx] = height    # h
    
    return cls_target, reg_target

def compute_loss(cls_pred, reg_pred, cls_target, reg_target, num_boxes):
    """
    Compute classification and regression losses
    Args:
        cls_pred: Classification predictions [B, H*W]
        reg_pred: Regression predictions [B, 4, H*W]
        cls_target: Classification targets [B, H*W]
        reg_target: Regression targets [B, 4, H*W]
        num_boxes: Number of valid boxes per image [B]
    """
    # Classification loss (Binary Cross Entropy) - penalize all grid cells
    cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='mean')

    # Regression loss (Smooth L1) - only for positive grid cells
    reg_loss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none')
    mask = (cls_target > 0).float()
    reg_loss = (reg_loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)

    return cls_loss, reg_loss

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Get data
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)
        event = batch['event'].to(device)
        boxes = batch['boxes'].to(device)
        num_boxes = batch['num_boxes'].to(device)
        
        # Forward pass
        cls_pred, reg_pred = model(rgb)
        
        # Calculate feature map size from predictions
        batch_size = cls_pred.shape[0]
        pred_size = cls_pred.shape[1]  # This should be H*W
        
        # Assume square feature map
        feature_size = (int(np.sqrt(pred_size)), int(np.sqrt(pred_size)))
        assert feature_size[0] * feature_size[1] == pred_size, f"Prediction size {pred_size} is not a perfect square"
        
        img_size = (rgb.shape[2], rgb.shape[3])
        
        # Generate targets
        cls_target, reg_target = generate_targets(boxes, feature_size, img_size)
        
        # Debug prints
        print(f"Prediction shape: {cls_pred.shape}")
        print(f"Target shape: {cls_target.shape}")
        print(f"Feature map size: {feature_size}")
        
        # Compute loss
        cls_loss, reg_loss = compute_loss(cls_pred, reg_pred, cls_target, reg_target, num_boxes)
        loss = cls_loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls_loss': f'{cls_loss.item():.4f}',
            'reg_loss': f'{reg_loss.item():.4f}'
        })
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_reg_loss

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            event = batch['event'].to(device)
            boxes = batch['boxes'].to(device)
            num_boxes = batch['num_boxes'].to(device)
            
            # Forward pass
            cls_pred, reg_pred = model(rgb)
            
            # Calculate feature map size from predictions
            batch_size = cls_pred.shape[0]
            pred_size = cls_pred.shape[1]  # This should be H*W
            feature_size = (int(np.sqrt(pred_size)), int(np.sqrt(pred_size)))
            
            # Ensure feature map size matches prediction size
            if feature_size[0] * feature_size[1] != pred_size:
                # If not a perfect square, use the actual dimensions
                feature_size = (pred_size, 1)  # Reshape to match prediction size
            
            img_size = (rgb.shape[2], rgb.shape[3])
            
            # Generate targets
            cls_target, reg_target = generate_targets(boxes, feature_size, img_size)
            
            # Compute loss
            cls_loss, reg_loss = compute_loss(cls_pred, reg_pred, cls_target, reg_target, num_boxes)
            loss = cls_loss + reg_loss
            
            # Update statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
            # Debug prints for each batch
            print(f"Batch loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Reg: {reg_loss.item():.4f})")
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    print(f"\nValidation Summary:")
    print(f"Total batches: {len(dataloader)}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average cls loss: {avg_cls_loss:.4f}")
    print(f"Average reg loss: {avg_reg_loss:.4f}")
    
    return avg_loss, avg_cls_loss, avg_reg_loss

def main():
    parser = argparse.ArgumentParser(description='Train vehicle detector')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TriAirDataset(args.data_root, transform=transform, split='train')
    val_dataset = TriAirDataset(args.data_root, transform=transform, split='val')
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    
    # Create model
    model = create_model(backbone_name=args.backbone, pretrained=True)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_cls_loss, val_reg_loss = validate(model, val_loader, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Reg: {val_reg_loss:.4f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Save best model
        print(f"\nCurrent best validation loss: {best_val_loss:.4f}")
        print(f"Current validation loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            print(f"Validation loss did not improve. Best loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main() 