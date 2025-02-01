import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from PIL import Image
from torchvision import models
from utils.config import IMAGE_SIZE, LABELS, THRESHOLD
from datasets.preprocess import resize_with_padding

thres = THRESHOLD
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ✅ 모델 로드 함수 (Teacher 또는 Student)
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
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(1280, num_classes)
        elif model_type == "student":
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(1024, num_classes)
        else:
            raise ValueError("model_type must be 'teacher' or 'student'")
        
        model.load_state_dict(torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
        models_list.append(model)
    return models_list


def predict_with_ensemble(image_path, models_list, class_labels, csv_file_path):
    """
    단일 이미지에 대해 앙상블 모델로 예측 수행
    Args:
        image_path (str): 입력 이미지 경로
        models_list (list): 앙상블에 사용될 모델 리스트
        class_labels (list): 클래스 이름 리스트
    Returns:
        tuple: 예측된 클래스 및 확률값
    """

    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    image = resize_with_padding(image, target_size=IMAGE_SIZE)  # Train과 동일한 전처리 적용
    image = image.unsqueeze(0).cuda() if torch.cuda.is_available() else image.unsqueeze(0)

    # 모든 모델의 예측 결과를 평균
    logits_sum = torch.zeros((1, len(class_labels))).cuda() if torch.cuda.is_available() else torch.zeros((1, len(class_labels)))
    with torch.no_grad():
        for model in models_list:
            logits = model(image)
            logits_sum += torch.nn.functional.softmax(logits, dim=1)  # Softmax로 확률 계산 후 합산

    # 평균 확률 계산
    logits_mean = logits_sum / len(models_list)
    predicted_class = torch.argmax(logits_mean, dim=1).item()
    confidence = logits_mean[0][predicted_class].item()
    
    if confidence >= thres:
        predicted_label = class_labels[predicted_class]
    else:
        predicted_label = "Uncertain"

    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        print(f"⚠️ Warning: {csv_file_path} 파일이 존재하지 않습니다. 새로 생성합니다.")
        df = pd.DataFrame(columns=["파일명", "정답 레이블", "예측 레이블", "확률"])

    # ✅ 기존 행에 예측 결과 추가 (파일명이 같은 경우 업데이트)
    df.loc[df["파일명"] == os.path.basename(image_path), ["예측 레이블", "확률"]] = [predicted_label, confidence]

    # ✅ CSV 저장 (덮어쓰기)
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    return class_labels[predicted_class], confidence



def infer_folder(folder_path, model, class_labels):
    """
    특정 폴더 내 모든 이미지에 대해 모델 추론 수행
    Args:
        folder_path (str): 이미지가 저장된 폴더 경로
        model (torch.nn.Module): 로드된 모델
        class_labels (list): 클래스 이름 리스트
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
    
    results = []  # 결과 저장 리스트

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        predicted_class, confidence = predict(image_path, model, class_labels)
        results.append((image_file, predicted_class, confidence))

    # ✅ 결과 출력
    print("\n===== Inference Results =====")
    for filename, pred_class, conf in results:
        print(f"{filename}: Predicted Class: {pred_class} (Confidence: {conf:.4f})")


def infer_folder_with_ensemble(folder_path, models_list, class_labels, csv_file_path):
    """
    특정 폴더 내 모든 이미지에 대해 앙상블 모델로 추론 수행
    Args:
        folder_path (str): 이미지가 저장된 폴더 경로
        models_list (list): 앙상블에 사용될 모델 리스트
        class_labels (list): 클래스 이름 리스트
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
    results = []  # 결과 저장 리스트

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        predicted_class, confidence = predict_with_ensemble(image_path, models_list, class_labels, csv_file_path)
        results.append((image_file, predicted_class, confidence))

    # ✅ 결과 출력
    print("\n===== Inference Results =====")
    for filename, pred_class, conf in results:
        print(f"{filename}: Predicted Class: {pred_class} (Confidence: {conf:.4f})")



def predict(image_path, model, class_labels, csv_file_path):
    """
    단일 이미지에 대한 모델 예측 수행
    Args:
        image_path (str): 입력 이미지 경로
        model (torch.nn.Module): 로드된 모델
        class_labels (list): 클래스 이름 리스트
    Returns:
        예측된 클래스 및 확률값
    """

    image = cv2.imread(image_path)
    image = resize_with_padding(image, target_size=IMAGE_SIZE)  # ✅ Train과 동일한 전처리 적용
    image = image.unsqueeze(0).cuda() if torch.cuda.is_available() else image.unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        output = model(image)  # 모델 예측
        probabilities = torch.nn.functional.softmax(output, dim=1)  # 확률값 변환
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        print(f"⚠️ Warning: {csv_file_path} 파일이 존재하지 않습니다. 새로 생성합니다.")
        df = pd.DataFrame(columns=["파일명", "정답 레이블", "예측 레이블", "확률"])

    # ✅ 기존 행에 예측 결과 추가 (파일명이 같은 경우 업데이트)
    df.loc[df["파일명"] == os.path.basename(image_path), ["예측 레이블", "확률"]] = [predicted_label, confidence]

    # ✅ CSV 저장 (덮어쓰기)
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    return predicted_label, confidence


# ✅ 실행 코드
if __name__ == "__main__":
    import os

    # 모델 타입 및 클래스 수 설정
    #model_type = "student"
    model_type = "teacher"

    class_labels = LABELS  # 클래스 레이블 (예: ["Class_0", "Class_1", ..., "Class_5"])
    num_classes = len(class_labels)

    # 테스트 데이터 폴더 및 클래스 레이블 설정
    test_folder_path = "/data/ephemeral/home/aims_image_classifier/data/test/"  # 테스트 이미지 폴더
    model_dir = "/data/ephemeral/home/aims_image_classifier/model_pth/"  # 모델 저장 폴더
    csv_file_path = f"/data/ephemeral/home/aims_image_classifier/data/inference_labels_{model_type}.csv"

    # 모델 경로 리스트 (앙상블용)
    model_paths = [
        os.path.join(model_dir, f"{model_type}_model_fold0.pth"),
        os.path.join(model_dir, f"{model_type}_model_fold1.pth"),
        os.path.join(model_dir, f"{model_type}_model_fold2.pth"),
    ]

    # ✅ 앙상블용 모델 로드
    models_list = load_models(model_paths, model_type=model_type, num_classes=num_classes)

    # ✅ 폴더 내 모든 이미지에 대해 앙상블 Inference 수행
    infer_folder_with_ensemble(test_folder_path, models_list, class_labels, csv_file_path)
