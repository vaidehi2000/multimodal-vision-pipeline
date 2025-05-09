import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        # Detection head with 3x3 convs and channel reduction
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        
        # Output layers for classification and regression
        self.cls_head = nn.Conv2d(in_channels//4, num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels//4, 4, 1)  # x, y, w, h
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Classification and regression outputs
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        
        return cls_out, reg_out

class VehicleDetector(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, num_classes=1):
        super().__init__()
        
        # Load backbone from timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1]  # Use last feature map
        )
        
        # Get number of channels from backbone
        in_channels = self.backbone.feature_info.channels()[-1]
        
        # Add spatial reduction layer to get square feature map
        self.spatial_reduction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Force 8x8 feature map (64 grid cells)
        )
        
        # Detection head
        self.detection_head = DetectionHead(in_channels, num_classes)
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)[0]
        
        # Apply spatial reduction
        features = self.spatial_reduction(features)
        
        # Get detection outputs
        cls_out, reg_out = self.detection_head(features)
        
        # Reshape outputs
        batch_size = x.shape[0]
        cls_out = cls_out.view(batch_size, -1)
        reg_out = reg_out.view(batch_size, 4, -1)
        
        return cls_out, reg_out

def create_model(backbone_name='resnet50', pretrained=True, num_classes=1):
    """Factory function to create the model"""
    return VehicleDetector(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes
    ) 