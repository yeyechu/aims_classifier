import os
import torch

import pandas as pd
from torchvision import models
from utils.config import IMAGE_SIZE, LABELS, THRESHOLD, BATCH_SIZE
from datasets.dataloader import get_data_loaders

from models.teacher_eff import CustomEfficientNet


def load_models(model_paths, model_type, num_classes):
    """
    모델 파일들을 로드하여 앙상블에 사용할 준비를 합니다.
    Args:
        model_paths (list): 모델 파일 경로 리스트
        model_type (str): 모델 종류 ("teacher" 또는 "student")
        num_classes (int): 클래스 개수
    Returns:
        list: 로드된 모델 리스트
    """
    models_list = []
    for path in model_paths:
        if model_type == "teacher":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            #model = CustomEfficientNet(num_classes=num_classes)
            #model.base_model.classifier[1] = torch.nn.Linear(1280, num_classes)
            model.classifier[1] = torch.nn.Linear(1280, num_classes)
        elif model_type == "student":
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            model.classifier[3] = torch.nn.Linear(1024, num_classes)
        else:
            raise ValueError("model_type must be 'teacher' or 'student'")
        
        model.load_state_dict(torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
        models_list.append(model)
    return models_list


def predict_with_ensemble(images, file_names, models_list, class_labels, csv_file_path):
    """
    배치 단위로 앙상블 모델로 예측 수행
    Args:
        images (Tensor): 배치 단위의 입력 이미지 텐서
        file_names (list): 해당 이미지들의 파일명 리스트
        models_list (list): 앙상블에 사용될 모델 리스트
        class_labels (list): 클래스 이름 리스트
        csv_file_path (str): 결과를 저장할 CSV 경로
    Returns:
        list: 예측된 클래스 및 확률값 리스트
    """
    logits_sum = torch.zeros((images.shape[0], len(class_labels))).cuda() if torch.cuda.is_available() else torch.zeros((images.shape[0], len(class_labels)))

    with torch.no_grad():
        for model in models_list:
            logits = model(images)
            logits_sum += torch.nn.functional.softmax(logits, dim=1)  # Softmax 후 확률 합산

    logits_mean = logits_sum / len(models_list)
    predicted_classes = torch.argmax(logits_mean, dim=1).cpu().numpy()
    confidences = logits_mean.max(dim=1)[0].cpu().numpy()

    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        print(f"⚠️ Warning: {csv_file_path} 파일이 존재하지 않습니다. 새로 생성합니다.")
        df = pd.DataFrame(columns=["파일명", "예측 레이블", "확률"])

    for file_name, pred_class, confidence in zip(file_names, predicted_classes, confidences):
        predicted_label = class_labels[pred_class]

        if file_name in df["파일명"].values:
            df.loc[df["파일명"] == file_name, ["예측 레이블", "확률"]] = [predicted_label, confidence]
        else:
            df = df.append({"파일명": file_name, "예측 레이블": predicted_label, "확률": confidence}, ignore_index=True)

    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    return predicted_classes, confidences


def infer_loader_with_ensemble(test_loader, models_list, class_labels, csv_file_path):
    """
    DataLoader를 활용하여 앙상블 모델로 추론 수행
    """
    results = []
    
    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.cuda() if torch.cuda.is_available() else images

            predicted_classes, confidences = predict_with_ensemble(images, file_names, models_list, class_labels, csv_file_path)
            
            for file_name, pred_class, confidence in zip(file_names, predicted_classes, confidences):
                results.append((file_name, pred_class, confidence))

    print("\n===== Inference Results =====")
    for filename, pred_class, conf in results:
        print(f"{filename}: Predicted Class: {class_labels[pred_class]} (Confidence: {conf:.4f})")



if __name__ == "__main__":
    import os

    # 모델 타입 및 클래스 수 설정
    model_type = "student"
    #model_type = "teacher"

    class_labels = LABELS
    batch_size = BATCH_SIZE
    num_classes = len(class_labels)

    test_folder_path = "/data/ephemeral/home/aims_image_classifier/data"
    model_dir = "/data/ephemeral/home/aims_image_classifier/model_pth/"
    csv_file_path = f"/data/ephemeral/home/aims_image_classifier/data/inference_labels_{model_type}.csv"

    model_paths = [
        os.path.join(model_dir, f"{model_type}_model_fold0.pth"),
        os.path.join(model_dir, f"{model_type}_model_fold1.pth"),
        os.path.join(model_dir, f"{model_type}_model_fold2.pth"),
        os.path.join(model_dir, f"{model_type}_model_fold3.pth"),
        os.path.join(model_dir, f"{model_type}_model_fold4.pth"),
    ]

    models_list = load_models(model_paths, model_type=model_type, num_classes=num_classes)
    test_loader = get_data_loaders(test_folder_path, batch_size)

    infer_loader_with_ensemble(test_loader, models_list, class_labels, csv_file_path)
