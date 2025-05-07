# models/vgg16.py
import torch.nn as nn
from torchvision import models

class VGG16MultiLabel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in self.backbone.features.parameters(): p.requires_grad_(False)
        in_feats = self.backbone.classifier[-1].in_features
        cls = list(self.backbone.classifier.children())
        cls[-1] = nn.Linear(in_feats, num_classes)
        self.backbone.classifier = nn.Sequential(*cls)
    def forward(self, x):
        return self.backbone(x)
