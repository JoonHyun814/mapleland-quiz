from torchvision import models
import torch.nn as nn

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=513):
        super().__init__()
        base = models.resnet50(pretrained=True)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)
