import torch
from torch import nn

class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model

        for p in self.text_model.parameters(): p.requires_grad = False
        for p in self.image_model.parameters(): p.requires_grad = False

        self.text_proj = nn.Sequential(nn.Linear(num_classes, 128), nn.ReLU(), nn.Dropout(0.3))
        self.image_proj = nn.Sequential(nn.Linear(num_classes, 128), nn.ReLU(), nn.Dropout(0.3))

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, text_input, image_input):
        text_logits = self.text_model(text_input)
        image_logits = self.image_model(image_input)
        text_feat = self.text_proj(text_logits)
        image_feat = self.image_proj(image_logits)
        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.classifier(combined)
