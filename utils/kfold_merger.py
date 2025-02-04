import os
import re
import torch
from torchvision import models
from config import NUM_CLASSES, K_FOLDS


def load_student_model(num_classes):
    """MobileNetV3 Small Student 모델 로드"""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    return model


def merge_kfold_models(model_dir, num_folds, num_classes, output_path):
    """K-Fold로 학습한 모델을 병합 (가중 평균)"""
    model_paths = [os.path.join(model_dir, f"student_model_fold{fold}.pth") for fold in range(num_folds)]
    merged_model = load_student_model(num_classes)

    merged_state_dict = None
    weight_sums = None

    # ✅ Validation Accuracy 파일 로드 (가중 평균을 위한 데이터)
    acc_file_path = os.path.join(model_dir, "validation_accuracies.txt")
    if os.path.exists(acc_file_path):
        with open(acc_file_path, "r") as f:
            fold_accuracies = []
            for line in f.readlines():
                match = re.search(r"Validation Accuracy:\s*([\d.]+)%", line)
                if match:
                    fold_accuracies.append(float(match.group(1))) 
    else:
        print("⚠️ Validation Accuracy 파일이 없어 단순 평균을 사용합니다.")
        fold_accuracies = [1.0] * num_folds

    total_weight = sum(fold_accuracies)

    for model_path, weight in zip(model_paths, fold_accuracies):
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        if merged_state_dict is None:
            merged_state_dict = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in state_dict.items()}
            weight_sums = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in state_dict.items()}

        for key in merged_state_dict:
            if "running_mean" in key or "running_var" in key:
                merged_state_dict[key] += state_dict[key] / num_folds
            else:
                merged_state_dict[key] += (state_dict[key] * weight)
                weight_sums[key] += weight

    for key in merged_state_dict:
        if "running_mean" not in key and "running_var" not in key:
            merged_state_dict[key] /= weight_sums[key].clamp(min=1e-8)

    merged_model.load_state_dict(merged_state_dict)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_model.state_dict(), output_path)
    print(f"가중 평균 적용된 앙상블 모델 저장 완료: {output_path}")


model_dir = "model_pth"
num_folds = K_FOLDS
num_classes = NUM_CLASSES
output_model_path = os.path.join(model_dir, "student_model_ensemble.pth")

merge_kfold_models(model_dir, num_folds, num_classes, output_model_path)
