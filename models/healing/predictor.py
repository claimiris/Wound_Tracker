import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class HealingPredictor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        base = resnet18(weights=weights)
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,x):
        x = self.backbone(x)
        x = self.fc(x)
        return x