import os
import cv2
import torch
import torch.nn as nn

import pandas as pd
import torch.nn.functional as F
from torchvision import models

from utils.config import IMAGE_SIZE, LABELS, THRESHOLD, BATCH_SIZE, NUM_CLASSES
from datasets.dataloader import get_data_loaders

from models.teacher_eff import CustomEfficientNet

thres = THRESHOLD

def load_single_model(model_path, model_type="student", num_classes=6):
    """
    병합된 모델을 로드하는 함수
    """
    if model_type == "teacher":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type == "student":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(1024, num_classes)
    else:
        raise ValueError("model_type은 'teacher' 또는 'student'여야 합니다.")

    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().cuda() if torch.cuda.is_available() else model.eval()
    return model



def infer_single_model(test_loader, model, class_labels, csv_file_path):
    """
    단일 모델을 사용하여 DataLoader에서 Inference 수행 및 결과 저장.
    
    Args:
        test_loader (DataLoader): 추론할 데이터 로더
        model (torch.nn.Module): 로드된 단일 모델
        class_labels (list): 클래스 이름 리스트
        csv_file_path (str): 결과를 저장할 CSV 경로
    """
    model.eval()
    results = []


    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        print(f"⚠️ Warning: {csv_file_path} 파일이 존재하지 않습니다. 새로 생성합니다.")
        df = pd.DataFrame(columns=["파일명", "예측 레이블", "확률"])

    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.cuda() if torch.cuda.is_available() else images

            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()

            for file_name, pred_class, confidence in zip(file_names, predicted_classes, confidences):
                predicted_label = class_labels[pred_class]

                if file_name in df["파일명"].values:
                    df.loc[df["파일명"] == file_name, ["예측 레이블", "확률"]] = [predicted_label, confidence]
                else:
                    df = df.append({"파일명": file_name, "예측 레이블": predicted_label, "확률": confidence}, ignore_index=True)

                results.append((file_name, predicted_label, confidence))

    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    print("\n===== Inference Results =====")
    for filename, pred_class, conf in results:
        print(f"{filename}: Predicted Class: {pred_class} (Confidence: {conf:.4f})")



if __name__ == "__main__":

    model_type = "student" # "teacher" or "student"

    class_labels = LABELS
    batch_size = BATCH_SIZE
    num_classes = NUM_CLASSES

    test_folder_path = "/data/ephemeral/home/aims_image_classifier/data"
    model_path = f"/data/ephemeral/home/aims_image_classifier/model_pth/{model_type}_model.pth"
    csv_file_path = f"/data/ephemeral/home/aims_image_classifier/data/inference_labels_{model_type}.csv"
    
    model = load_single_model(model_path, model_type=model_type, num_classes=num_classes)
    test_loader = get_data_loaders(test_folder_path, batch_size)

    infer_single_model(test_loader, model, class_labels, csv_file_path)
    
