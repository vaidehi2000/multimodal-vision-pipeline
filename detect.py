import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from model import create_model
from dataset import TriAirDataset, collate_fn

def non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.25):
    """Perform non-maximum suppression on predicted boxes"""
    # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # Initialize list of picked indexes
    keep = []
    
    # Sort by confidence score
    idxs = torch.argsort(scores, descending=True)
    
    while len(idxs) > 0:
        # Pick the box with highest confidence
        current = idxs[0]
        keep.append(current.item())
        
        if len(idxs) == 1:
            break
            
        # Remove the current box
        idxs = idxs[1:]
        
        # Get coordinates of remaining boxes
        xx1 = torch.max(x1[current], x1[idxs])
        yy1 = torch.max(y1[current], y1[idxs])
        xx2 = torch.min(x2[current], x2[idxs])
        yy2 = torch.min(y2[current], y2[idxs])
        
        # Calculate intersection area
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        intersection = w * h
        
        # Calculate areas
        area1 = (x2[current] - x1[current]) * (y2[current] - y1[current])
        area2 = (x2[idxs] - x1[idxs]) * (y2[idxs] - y1[idxs])
        
        # Calculate IoU
        iou = intersection / (area1 + area2 - intersection)
        
        # Remove boxes with IoU > threshold
        idxs = idxs[iou <= iou_threshold]
    
    return torch.tensor(keep)

def process_predictions(cls_pred, reg_pred, conf_threshold=0.25, iou_threshold=0.5):
    """Process raw model predictions to get final detections"""
    batch_size = cls_pred.shape[0]
    detections = []
    
    # Debug prints
    print(f"\nPrediction shapes - cls: {cls_pred.shape}, reg: {reg_pred.shape}")
    print(f"Confidence threshold: {conf_threshold}")
    
    for b in range(batch_size):
        # Get predictions for this image
        cls = cls_pred[b]
        reg = reg_pred[b]
        
        # Get confidence scores
        scores = torch.sigmoid(cls)
        
        # Debug prints for first batch
        if b == 0:
            print(f"\nFirst image stats:")
            print(f"Max confidence: {scores.max().item():.4f}")
            print(f"Mean confidence: {scores.mean().item():.4f}")
            print(f"Number of predictions above threshold: {(scores > conf_threshold).sum().item()}")
        
        # Filter by confidence
        mask = scores > conf_threshold
        if not mask.any():
            detections.append([])
            continue
            
        # Get filtered boxes and scores
        boxes = reg[:, mask].T  # [N, 4]
        scores = scores[mask]   # [N]
        
        # Debug prints for first batch with detections
        if b == 0 and len(boxes) > 0:
            print(f"\nFirst detection stats:")
            print(f"Number of boxes after confidence filtering: {len(boxes)}")
            print(f"Box coordinates range:")
            print(f"x: [{boxes[:, 0].min().item():.4f}, {boxes[:, 0].max().item():.4f}]")
            print(f"y: [{boxes[:, 1].min().item():.4f}, {boxes[:, 1].max().item():.4f}]")
            print(f"w: [{boxes[:, 2].min().item():.4f}, {boxes[:, 2].max().item():.4f}]")
            print(f"h: [{boxes[:, 3].min().item():.4f}, {boxes[:, 3].max().item():.4f}]")
        
        # Apply NMS
        keep = non_max_suppression(boxes, scores, iou_threshold, conf_threshold)
        
        # Get final detections
        if len(keep) > 0:
            final_boxes = boxes[keep]
            final_scores = scores[keep]
            
            # Convert to list of [x, y, w, h, score]
            dets = torch.cat([final_boxes, final_scores.unsqueeze(1)], dim=1)
            detections.append(dets.tolist())
        else:
            detections.append([])
    
    return detections

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in (x, y, w, h) format"""
    # If box2 includes class_id, extract just the box coordinates
    if len(box2) == 5:  # [class_id, x, y, w, h]
        box2 = box2[1:]  # Remove class_id
    
    # Convert to (x1, y1, x2, y2) format
    x1 = box1[0] - box1[2] / 2
    y1 = box1[1] - box1[3] / 2
    x2 = box1[0] + box1[2] / 2
    y2 = box1[1] + box1[3] / 2
    
    x3 = box2[0] - box2[2] / 2
    y3 = box2[1] - box2[3] / 2
    x4 = box2[0] + box2[2] / 2
    y4 = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)
    
    if x6 < x5 or y6 < y5:
        return 0.0
    
    intersection = (x6 - x5) * (y6 - y5)
    
    # Calculate areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # Calculate IoU
    iou = intersection / (area1 + area2 - intersection)
    return iou

def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, and mAP"""
    all_precisions = []
    all_recalls = []
    all_aps = []
    
    print("\nCalculating metrics:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of ground truth: {len(ground_truth)}")
    
    for pred_boxes, gt_boxes in zip(predictions, ground_truth):
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue
            
        # Convert predictions to numpy array
        pred_boxes = np.array(pred_boxes)
        gt_boxes = np.array(gt_boxes)
        
        print(f"\nProcessing image:")
        print(f"Number of predictions: {len(pred_boxes)}")
        print(f"Number of ground truth boxes: {len(gt_boxes)}")
        
        if len(pred_boxes) == 0:
            precision = 0.0
            recall = 0.0
            ap = 0.0
        else:
            # Sort predictions by confidence
            scores = pred_boxes[:, -1]
            sorted_idx = np.argsort(-scores)
            pred_boxes = pred_boxes[sorted_idx]
            
            # Initialize arrays for tracking matches
            gt_matched = np.zeros(len(gt_boxes))
            pred_matched = np.zeros(len(pred_boxes))
            
            # Match predictions to ground truth
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if gt_matched[j]:
                        continue
                        
                    iou = calculate_iou(pred_box[:4], gt_box)  # gt_box includes class_id
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = 1
                    pred_matched[i] = 1
            
            # Calculate precision and recall
            true_positives = np.sum(pred_matched)
            precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
            recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
            
            print(f"True positives: {true_positives}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            # Calculate AP
            ap = precision * recall
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_aps.append(ap)
    
    # Calculate mean metrics
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_ap = np.mean(all_aps)
    
    print(f"\nFinal metrics:")
    print(f"Mean precision: {mean_precision:.4f}")
    print(f"Mean recall: {mean_recall:.4f}")
    print(f"Mean AP: {mean_ap:.4f}")
    
    return {
        'precision': mean_precision,
        'recall': mean_recall,
        'mAP': mean_ap
    }

def visualize_detections(image, predictions, ground_truth, save_path):
    """
    Visualize predictions and ground truth boxes on an image
    Args:
        image: RGB image as numpy array
        predictions: List of [x, y, w, h, score] for predictions
        ground_truth: List of [class_id, x, y, w, h] for ground truth boxes
        save_path: Path to save visualization
    """
    # Convert image to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Debug prints
    print(f"\nImage shape: {image.shape}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of ground truth boxes: {len(ground_truth)}")
    
    # Draw predictions
    for i, pred in enumerate(predictions):
        x, y, w, h, score = pred
        
        # Convert normalized coordinates to pixel coordinates
        img_h, img_w = image.shape[:2]
        x1 = int((x - w/2) * img_w)
        y1 = int((y - h/2) * img_h)
        x2 = int((x + w/2) * img_w)
        y2 = int((y + h/2) * img_h)
        
        # Debug print for first prediction
        if i == 0:
            print("\nFirst prediction:")
            print(f"Raw coordinates: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}, score={score:.4f}")
            print(f"Pixel coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw score
        score_text = f"{score:.2f}"
        cv2.putText(image, score_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ground truth
    for i, gt in enumerate(ground_truth):
        class_id, x, y, w, h = gt  # Unpack class_id along with box coordinates
        
        # Convert normalized coordinates to pixel coordinates
        img_h, img_w = image.shape[:2]
        x1 = int((x - w/2) * img_w)
        y1 = int((y - h/2) * img_h)
        x2 = int((x + w/2) * img_w)
        y2 = int((y + h/2) * img_h)
        
        # Debug print for first ground truth
        if i == 0:
            print("\nFirst ground truth:")
            print(f"Raw coordinates: class={class_id}, x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
            print(f"Pixel coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw class ID
        class_text = f"Class {int(class_id)}"
        cv2.putText(image, class_text, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save image
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def run_inference(model, dataloader, device, conf_threshold=0.25, iou_threshold=0.5):
    """Run inference on a dataset"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            # Get data
            rgb = batch['rgb'].to(device)
            
            # Forward pass
            cls_pred, reg_pred = model(rgb)
            
            # Process predictions
            detections = process_predictions(
                cls_pred, reg_pred,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            all_predictions.extend(detections)
    
    return all_predictions

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

def main():
    parser = argparse.ArgumentParser(description='Run detection with inference and evaluation modes')
    parser.add_argument('--mode', type=str, choices=['inference', 'evaluate'], required=True,
                      help='Mode to run: inference or evaluate')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='IoU threshold for NMS (default: 0.3)')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TriAirDataset(args.data_root, transform=transform, split='test')
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn  # Add the custom collate function
    )
    
    # Run inference
    predictions = run_inference(
        model, dataloader, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    if args.mode == 'inference':
        # Print predictions
        for i, dets in enumerate(predictions):
            print(f"\nImage {i}:")
            for det in dets:
                x, y, w, h, score = det
                print(f"Box: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), Score: {score:.2f}")
    
    else:  # evaluate mode
        # Get ground truth boxes
        ground_truth = []
        for i in range(len(dataset)):
            sample = dataset[i]
            ground_truth.append(sample['boxes'].numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth, args.iou_threshold)
        print("\nEvaluation Metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"mAP: {metrics['mAP']:.4f}")
        
        # Visualize detections
        print("\nGenerating visualizations...")
        for i, (pred, gt) in enumerate(tqdm(zip(predictions, ground_truth))):
            # Get original image
            sample = dataset[i]
            image = sample['rgb'].permute(1, 2, 0).numpy()
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            image = image.astype(np.uint8)
            
            # Visualize
            save_path = os.path.join(args.save_dir, f'detection_{i:04d}.png')
            visualize_detections(image, pred, gt, save_path)
        
        print(f"\nVisualizations saved to {args.save_dir}")

if __name__ == '__main__':
    main() 