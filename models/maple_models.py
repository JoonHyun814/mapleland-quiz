from torchvision import models
import torch.nn as nn
import timm

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=513):
        super().__init__()
        base = models.resnet50(pretrained=True)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=513):
        super().__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        
        # 기존 분류기 레이어 교체
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
