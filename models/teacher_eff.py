import torch.nn as nn
from torchvision import models

# 기존 모델의 Fully Connected Layer에 Dropout 추가
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class FeatureExtractorWrapper(nn.Module):
    """
    기존 EfficientNet을 감싸고 Feature를 반환할 수 있도록 수정하는 Wrapper 클래스
    """
    def __init__(self, base_model):
        super(FeatureExtractorWrapper, self).__init__()
        self.feature_extractor = base_model.features  # ✅ EfficientNet의 Feature 추출 부분
        self.pooling = base_model.avgpool  # ✅ Global Average Pooling
        self.classifier = base_model.classifier  # ✅ 원래의 분류 레이어 (Linear)

    def forward(self, x):
        features = self.feature_extractor(x)  # 🔹 중간 Feature 추출
        pooled_features = self.pooling(features).flatten(1)  # 🔹 Flatten

        logits = self.classifier(pooled_features)  # 🔹 최종 Logits
        return logits, pooled_features  # 🔹 Logits와 Feature 반환
