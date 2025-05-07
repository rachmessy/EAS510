import torch.nn as nn
from torchvision import models

class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5, freeze_backbone=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad_(False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)