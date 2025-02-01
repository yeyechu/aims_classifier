import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp  # Mixed Precision Training
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.eda.set_font_matplot import set_nanumgothic_font
set_nanumgothic_font()

from loss import DistillationLoss
from utils.early_stopping import EarlyStopping
from utils.config import LABELS


def train_teacher_student(teacher, student, train_loader, val_loader, epochs, lr, temperature, alpha):
    criterion = DistillationLoss(temperature, alpha)
    optimizer = optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                teacher_logits = teacher(images)  # Teacher의 예측

            student_logits = student(images)  # Student의 예측
            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        validate(student, val_loader)


def train_teacher(teacher, train_loader, val_loader, epochs, lr, num_classes, fold_idx):
    teacher.classifier[1] = nn.Linear(1280, num_classes)  # EfficientNet
    teacher = teacher.train().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=lr)
    scaler = amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    best_val_loss = float("inf")  # Best Validation Loss 초기화

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        teacher.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            with amp.autocast("cuda"):
                outputs = teacher(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(teacher, val_loader, LABELS)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early Stopping 적용! 학습 중단")
            break


        # ✅ 모델 저장 디렉토리 생성
        model_dir = "model_pth"
        os.makedirs(model_dir, exist_ok=True)

        # ✅ Fold별 Teacher 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(model_dir, f"teacher_model_fold{fold_idx}.pth")
            torch.save(teacher.state_dict(), model_save_path)
            print(f"✅ Best Model Updated: {model_save_path} (Validation Loss: {best_val_loss:.4f})")
    


# 모델 검증 함수
def validate(model, val_loader, class_labels, save_dir="validation_visualization"):
    """
    모델 검증 함수 (정확도, 손실 계산 + 예측 결과 이미지에 한글 텍스트 출력 후 저장)
    
    Args:
        model (torch.nn.Module): 검증할 모델
        val_loader (DataLoader): 검증 데이터 로더
        class_labels (list): 클래스 이름 리스트
        save_dir (str): 시각화된 이미지를 저장할 폴더 경로
    """
    # ✅ 저장할 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 한글 폰트 설정 (폰트 경로 확인 필요)
    font_path = "/data/ephemeral/home/aims_image_classifier/utils/eda/fonts/NanumGothicCoding.ttf"
    font = ImageFont.truetype(font_path, 30)

    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            for idx in range(images.shape[0]):
                true_label = labels[idx].cpu().item()
                pred_label = preds[idx].cpu().item()
                true_class = class_labels[true_label]
                pred_class = class_labels[pred_label]

                # ✅ 이미지 변환 (정규화 해제)
                image_np = images[idx].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)  # 정규화 해제

                # ✅ OpenCV에서 PIL로 변환
                image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)

                # ✅ 한글 텍스트 삽입
                color = (0, 255, 0) if true_class == pred_class else (255, 0, 0)  # 맞으면 초록, 틀리면 빨강
                draw.text((10, 30), f"정답: {true_class} | 예측: {pred_class}", font=font, fill=color)

                # ✅ PIL 이미지를 다시 OpenCV 형식으로 변환
                annotated_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                # ✅ 이미지 저장
                save_path = os.path.join(save_dir, f"val_{batch_idx}_{idx}.png")
                cv2.imwrite(save_path, annotated_image)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    print(f"\nValidation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}")

    return avg_loss


# MobileNet Student 모델 학습 함수 (지식 증류 적용)
def train_student(teacher, student, train_loader, val_loader, epochs, lr, temperature, alpha, num_classes, fold_idx):
    """
    K-Fold 교차 검증을 적용한 Student 모델 학습 함수.

    Args:
        teacher (nn.Module): Teacher 모델
        student (nn.Module): MobileNet Student 모델
        data_dir (str): 데이터 디렉토리 경로
        batch_size (int): 배치 크기
        epochs (int): 학습할 Epoch 수
        lr (float): Learning Rate
        temperature (float): 지식 증류 온도 값
        alpha (float): 지식 증류 가중치
        num_classes (int): 클래스 개수
        fold_idx (int): 현재 K-Fold의 인덱스 (Fold 번호)
        n_splits (int): K-Fold 개수
        random_seed (int): 랜덤 시드 값
    """
    # ✅ 모델 저장 디렉토리 생성
    model_dir = "model_pth"
    os.makedirs(model_dir, exist_ok=True)

    teacher_model_path = os.path.join(model_dir, f"teacher_model_fold{fold_idx}.pth")
    teacher.load_state_dict(torch.load(teacher_model_path))
    teacher = teacher.eval().cuda()

    student.classifier[3] = nn.Linear(1024, num_classes)  # MobileNet
    student = student.train().cuda()

    criterion = DistillationLoss(temperature, alpha)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scaler = amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    best_val_loss = float("inf")  # Best Validation Loss 초기화

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        student.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                teacher_logits = teacher(images)

            with amp.autocast("cuda"):
                student_logits = student(images)
                loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(student, val_loader, LABELS)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early Stopping 적용! 학습 중단")
            break
    

        # ✅ Fold별 Student 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(model_dir, f"student_model_fold{fold_idx}.pth")
            torch.save(student.state_dict(), model_save_path)
            print(f"✅ Best Model Updated: {model_save_path} (Validation Loss: {best_val_loss:.4f})")

