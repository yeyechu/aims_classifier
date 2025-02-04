import torch.nn as nn
from torchvision import models


class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomMobileNet, self).__init__()
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # ✅ Feature 추출을 위한 레이어 저장
        self.feature_extractor = self.model.features
        self.pooling = self.model.avgpool
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # 🔹 중간 Feature 추출
        pooled_features = self.pooling(features).flatten(1)  # 🔹 Flatten

        logits = self.classifier(pooled_features)  # 🔹 최종 Logits
        return logits, pooled_features  # 🔹 Logits와 Feature 반환
