import os
import cv2
import torch
import torch.nn as nn

import pandas as pd
import torch.nn.functional as F
from torchvision import models

from utils.config import IMAGE_SIZE, LABELS, THRESHOLD, BATCH_SIZE, NUM_CLASSES
from datasets.dataloader import get_test_data_loaders

from models.teacher_eff import CustomEfficientNet
import wandb
import time

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
        # model = models.shufflenet_v2_x1_0(weights=None)
        # model.fc = nn.Linear(1024, num_classes)
    else:
        raise ValueError("model_type은 'teacher' 또는 'student'여야 합니다.")

    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().cuda() if torch.cuda.is_available() else model.eval()
    return model


def infer_single_model(test_loader, model, class_labels, csv_file_path, model_type, device="cuda"):
    """
    단일 모델을 사용하여 DataLoader에서 Inference 수행 및 결과 저장.
    - ✅ GPU 연산 동기화하여 정확한 추론 속도 측정
    - ✅ WandB Table을 활용하여 Inference 결과 로깅
    - ✅ CSV 파일에 예측 결과와 Latency 저장

    Args:
        test_loader (DataLoader): 추론할 데이터 로더
        model (torch.nn.Module): 로드된 단일 모델
        class_labels (list): 클래스 이름 리스트
        csv_file_path (str): 결과를 저장할 CSV 경로
        model_type (str): "teacher" 또는 "student" (WandB 로깅 구분용)
        device (str): "cuda" 또는 "cpu"
    """
    model.to(device)
    model.eval()
    results = []

    # ✅ WandB 세션 시작
    wandb.init(project="document_classification", name=f"Inference_{model_type}", group="Inference")
    table = wandb.Table(columns=["파일명", "예측 레이블", "확률", "추론 시간(ms)"])

    # ✅ CSV 파일 로드 또는 새로 생성
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        print(f"⚠️ Warning: {csv_file_path} 파일이 존재하지 않습니다. 새로 생성합니다.")
        df = pd.DataFrame(columns=["파일명", "예측 레이블", "확률", "추론 시간(ms)"])

    # ✅ 실시간 추론 속도 측정 (Batch 기반 Latency 측정)
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.no_grad():
        for _ in range(10):  # ✅ Warm-up (GPU 최적화)
            _ = model(x)

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        _ = model(x)
        end_time.record()

        torch.cuda.synchronize()  # ✅ GPU 연산 동기화
        latency = start_time.elapsed_time(end_time)  # ms 단위
        print(f"\n🚀 실시간 추론 속도: {latency:.3f} ms (1 이미지당)\n")

    # ✅ Inference 수행
    total_latency = 0
    total_images = 0

    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.to(device)

            batch_start_time = torch.cuda.Event(enable_timing=True)
            batch_end_time = torch.cuda.Event(enable_timing=True)

            batch_start_time.record()
            outputs = model(images)
            batch_end_time.record()

            torch.cuda.synchronize()  # ✅ GPU 연산 동기화
            batch_latency = batch_start_time.elapsed_time(batch_end_time)  # ms 단위

            # ✅ 평균 Latency 계산을 위해 누적
            total_latency += batch_latency
            total_images += len(file_names)

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()

            for file_name, pred_class, confidence in zip(file_names, predicted_classes, confidences):
                predicted_label = class_labels[pred_class]

                # ✅ CSV 파일 업데이트
                if file_name in df["파일명"].values:
                    df.loc[df["파일명"] == file_name, ["예측 레이블", "확률", "추론 시간(ms)"]] = [predicted_label, confidence, batch_latency / len(file_names)]
                else:
                    df = df.append({"파일명": file_name, "예측 레이블": predicted_label, "확률": confidence, "추론 시간(ms)": batch_latency / len(file_names)}, ignore_index=True)

                # ✅ WandB Table에 저장
                table.add_data(file_name, predicted_label, confidence, batch_latency / len(file_names))

                results.append((file_name, predicted_label, confidence, batch_latency / len(file_names)))


    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
    wandb.log({"Inference Results": table, "Average Latency (ms)": latency})
    

    answer = ["체력평가", "생활기록부대체양식", "주민등록본", "국민체력100", "기초생활수급자증명서",
              "국민체력100", "검정고시합격자증명서", "체력평가", "생활기록부대체양식", "검정고시합격자증명서",
              "기초생활수급자증명서", "주민등록본", "검정고시합격자증명서"]
    idx = 0

    for filename, pred_class, conf, latency in results:
        print(f"{filename}: {answer[idx]} → {pred_class} (Confidence: {conf:.4f}, Latency: {latency:.3f} ms)")
        idx += 1

    wandb.finish()



if __name__ == "__main__":

    type_list = ["teacher", "student"]
    model_type = type_list[1] # 0: "teacher", 1: "student"
    print(f"\n===== Inference Results(model type: {model_type}) =====")

    class_labels = LABELS
    batch_size = 1
    num_classes = NUM_CLASSES

    test_folder_path = "/data/ephemeral/home/aims_image_classifier/data"
    model_path = f"/data/ephemeral/home/aims_image_classifier/model_pth/{model_type}_model.pth"
    csv_file_path = f"/data/ephemeral/home/aims_image_classifier/data/inference_labels_{model_type}.csv"
    
    model = load_single_model(model_path, model_type=model_type, num_classes=num_classes)
    test_loader = get_test_data_loaders(test_folder_path, batch_size)

    infer_single_model(test_loader, model, class_labels, csv_file_path, model_type)
    
