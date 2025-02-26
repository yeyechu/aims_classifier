import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from config import LABELS, IMAGE_SIZE

def load_model(model_path, num_classes=6, device="cuda"):
    """
    학습된 MobileNetV3-Small 모델을 로드하는 함수
    """
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, num_classes)

    # model = models.efficientnet_b0(weights=None)
    # model.classifier[1] = nn.Linear(1280, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def preprocess_image(image_path, device="cuda"):
    """
    이미지를 로드하고 전처리하는 함수
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 후 이동
    return image

def infer_and_visualize(model, image, class_names, feature_map_dir):
    """
    모델을 사용하여 추론을 수행하고 Feature Map을 저장 및 시각화하는 함수
    """
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    num = 0
    hook_layer = model.features[num]  # 특정 레이어 선택
    hook_handle = hook_layer.register_forward_hook(get_activation(f"features.{num}"))
    
    # 모델 추론 수행
    with torch.no_grad():
        outputs = model(image)
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).cpu().item()
    confidence = probabilities.max().cpu().item()
    
    # Feature Map 저장 및 시각화
    feature_map = activation[f"features.{num}"].squeeze()
    os.makedirs(feature_map_dir, exist_ok=True)
    feature_map_path = os.path.join(feature_map_dir, "feature_map.npy")
    np.save(feature_map_path, feature_map)
    
    feature_map_image_path = os.path.join(feature_map_dir, "feature_map.png")
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < feature_map.shape[0]:
            ax.imshow(feature_map[i], cmap="viridis")
            ax.axis("off")
    plt.savefig(feature_map_image_path)
    plt.close()
    
    return class_names[predicted_class], confidence

# 실행 예제
if __name__ == "__main__":
    model_path = "/data/ephemeral/home/aims_image_classifier/model_pth/student_model.pth"
    #model_path = "/data/ephemeral/home/aims_image_classifier/model_pth/teacher_model.pth"

    image_path = "/data/ephemeral/home/aims_image_classifier/data/test2/image_071.png"
    feature_map_dir = "/data/ephemeral/home/aims_image_classifier/feature_maps"
    class_names = LABELS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=len(class_names), device=device)
    image = preprocess_image(image_path, device=device)
    predicted_label, confidence = infer_and_visualize(model, image, class_names, feature_map_dir)
    
    print(f"🔍 예측 결과: {predicted_label} (Confidence: {confidence:.4f})")