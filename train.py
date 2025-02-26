import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp  # Mixed Precision Training
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.eda.set_font_matplot import set_nanumgothic_font
set_nanumgothic_font()

from loss import DistillationLoss, FeatureDistillationLoss
from utils.early_stopping import EarlyStopping
from utils.config import LABELS, EPOCHES, LEARNING_RATE, BATCH_SIZE

import wandb


def train_teacher(teacher, train_loader, val_loader, epochs, lr, num_classes, fold_idx, patience):
    #teacher.base_model.classifier[1] = nn.Linear(1280, num_classes)
    teacher.classifier[1] = nn.Linear(1280, num_classes)
    teacher = teacher.train().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=lr)
    scaler = amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=patience)

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

        model_dir = "model_pth"
        os.makedirs(model_dir, exist_ok=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(model_dir, f"teacher_model_fold{fold_idx}.pth")
            torch.save(teacher.state_dict(), model_save_path)
            print(f"Best Model Updated: {model_save_path} (Validation Loss: {best_val_loss:.4f})")

        if early_stopping.early_stop:
            print("Early Stopping 적용! 학습 중단")
            break
    


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

                image_np = images[idx].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)  # 정규화 해제

                image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)

                color = (0, 255, 0) if true_class == pred_class else (255, 0, 0)  # 맞으면 초록, 틀리면 빨강
                draw.text((10, 30), f"정답: {true_class} | 예측: {pred_class}", font=font, fill=color)

                annotated_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                save_path = os.path.join(save_dir, f"val_{batch_idx}_{idx}.png")
                cv2.imwrite(save_path, annotated_image)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    print(f"\nValidation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}")

    return avg_loss


# MobileNet Student 모델 학습 함수 (지식 증류 적용)
def train_student(teacher, student, train_loader, val_loader, epochs, lr, temperature, alpha, num_classes, fold_idx, patience):
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

    # teacher_model_path = os.path.join(model_dir, f"teacher_model_fold{fold_idx}.pth")
    # teacher.load_state_dict(torch.load(teacher_model_path))
    teacher = teacher.eval().cuda()

    student.classifier[3] = nn.Linear(1024, num_classes)  # MobileNet
    student = student.train().cuda()

    criterion = DistillationLoss(temperature, alpha)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scaler = amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=patience)

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
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(model_dir, f"student_model_fold{fold_idx}.pth")
            torch.save(student.state_dict(), model_save_path)
            print(f"Best Model Updated: {model_save_path} (Validation Loss: {best_val_loss:.4f})")

        if early_stopping.early_stop:
            print("Early Stopping 적용! 학습 중단")
            break


# Feature Distillation용 MSE Loss
criterion_feature_distill = nn.MSELoss()


def get_intermediate_features(model, layer_name):
    """
    특정 레이어의 Feature Map을 가져오기 위한 Hook 등록 함수
    """
    features = {}

    def hook(module, input, output):
        features[layer_name] = output

    layer = dict([*model.named_modules()])[layer_name]  # 원하는 레이어 선택
    layer.register_forward_hook(hook)
    
    return features


def train_student_with_fd(teacher, student, train_loader, val_loader, epochs, lr, temperature, alpha, num_classes, fold_idx, patience):
    model_dir = "model_pth"
    os.makedirs(model_dir, exist_ok=True)

    teacher_model_path = os.path.join(model_dir, f"teacher_model_fold{fold_idx}.pth")
    teacher.load_state_dict(torch.load(teacher_model_path))
    teacher = teacher.eval().cuda()

    student.classifier[3] = nn.Linear(1024, num_classes)  # MobileNet
    student = student.train().cuda()

    # ✅ Feature Distillation Loss 추가
    criterion_feature = FeatureDistillationLoss(alpha=0.7, temperature=4.0, feature_weight=0.2)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    scaler = amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        student.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                teacher_logits, teacher_features_out = teacher(images)

            with amp.autocast("cuda"):
                student_logits, student_features_out = student(images)

                # ✅ Feature Distillation Loss 계산
                loss_fd = criterion_feature(student_logits, teacher_logits, student_features_out, teacher_features_out, labels)

            optimizer.zero_grad()
            scaler.scale(loss_fd).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_fd.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(student, val_loader, LABELS)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(model_dir, f"student_model_fold{fold_idx}.pth")
            torch.save(student.state_dict(), model_save_path)
            print(f"✅ Best Model Updated: {model_save_path} (Validation Loss: {best_val_loss:.4f})")

        if early_stopping.early_stop:
            print("Early Stopping 적용! 학습 중단")
            break


def validate_va(model, val_loader, class_labels):
    """
    모델 검증 함수
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader) 

    print(f"Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}")

    return avg_loss, accuracy 


def train_student_with_va(teacher, student, train_loader, val_loader, epochs, lr, temperature, alpha, num_classes, fold_idx, patience):
    """
    K-Fold 교차 검증을 적용한 Student 모델 학습 함수.

    Args:
        teacher (nn.Module): Teacher 모델
        student (nn.Module): MobileNet Student 모델
        train_loader (DataLoader): 학습 데이터 로더
        val_loader (DataLoader): 검증 데이터 로더
        epochs (int): 학습할 Epoch 수
        lr (float): Learning Rate
        temperature (float): 지식 증류 온도 값
        alpha (float): 지식 증류 가중치
        num_classes (int): 클래스 개수
        fold_idx (int): 현재 K-Fold의 인덱스 (Fold 번호)
        patience (int): Early Stopping patience 값
    """

    model_dir = "model_pth"
    os.makedirs(model_dir, exist_ok=True)

    val_acc_file = os.path.join(model_dir, f"validation_accuracies.txt")

    teacher = teacher.eval().cuda()

    student.classifier[3] = torch.nn.Linear(1024, num_classes)
    student = student.train().cuda()

    criterion = DistillationLoss(temperature, alpha)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scaler = amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_val_acc = 0.0

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
        val_loss, val_acc = validate_va(student, val_loader, LABELS)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        with open(val_acc_file, "a") as f:
            f.write(f"Fold {fold_idx}, Epoch {epoch+1}, Validation Accuracy: {val_acc:.2f}%\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            model_save_path = os.path.join(model_dir, f"student_model_fold{fold_idx}.pth")
            torch.save(student.state_dict(), model_save_path)
            print(f"✅ Best Model Updated: {model_save_path} (Validation Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.2f}%)")

        if early_stopping.early_stop:
            print("✅ Early Stopping 적용! 학습 중단")
            break


def train_teacher_all(teacher, train_loader, epochs, lr, num_classes, patience):
    """
    Teacher 모델을 학습하는 함수 (Validation 없이 Train만 수행)

    Args:
        teacher (nn.Module): Teacher 모델
        epochs (int): 학습할 Epoch 수
        lr (float): Learning Rate
        num_classes (int): 클래스 개수
    """

    teacher.classifier[1] = nn.Linear(1280, num_classes)  # EfficientNet
    teacher = teacher.train().cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = amp.GradScaler()

    model_dir = "model_pth"
    os.makedirs(model_dir, exist_ok=True)

    best_train_loss = float("inf")  # Best Train Loss 초기화

    wandb.init(
    project="document_classification",
    name="EfficientNet-B0",
    group="Teacher",
    config={
        "epochs": EPOCHES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__
    }
)
    patience_counter = 0
    early_stopping_patience = patience

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

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        scheduler.step(avg_train_loss)
        #scheduler.step()
        
        wandb.log({"Techer Train Loss": avg_train_loss, "Techer Learning Rate": optimizer.param_groups[0]['lr'], "Techer Epoch": epoch + 1})

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            model_save_path = os.path.join(model_dir, f"teacher_model.pth")
            torch.save(teacher.state_dict(), model_save_path)
            print(f"Best Model Updated: {model_save_path} (Train Loss: {best_train_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early Stopping")
            break
    
    wandb.finish()


# def train_student_all(teacher, student, train_loader, epochs, lr, temperature, alpha, num_classes, patience):
#     """
#     Teacher 모델을 활용하여 Student 모델을 학습하는 함수 (Validation 없이 Train만 수행)

#     Args:
#         teacher (nn.Module): Teacher 모델
#         student (nn.Module): MobileNet Student 모델
#         train_loader (DataLoader): Train 데이터 로더
#         epochs (int): 학습할 Epoch 수
#         lr (float): Learning Rate
#         temperature (float): 지식 증류 온도 값
#         alpha (float): 지식 증류 가중치
#         num_classes (int): 클래스 개수
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model_dir = "model_pth"
#     os.makedirs(model_dir, exist_ok=True)

#     teacher = teacher.to(device).eval()

#     student.classifier[3] = nn.Linear(1024, num_classes)  # MobileNet
#     student = student.to(device).train()

#     from utils.pruning import apply_pruning
#     student = apply_pruning(student, amount=0.3)

#     criterion = DistillationLoss(temperature, alpha)
#     optimizer = optim.Adam(student.parameters(), lr=lr)
#     scaler = amp.GradScaler()
#     scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

#     student = torch.quantization.quantize_dynamic(student, {nn.Linear}, dtype=torch.qint8)

#     # Early Stopping 변수
#     best_train_loss = float("inf")
#     patience_counter = 0
#     early_stopping_patience = patience

#     for epoch in range(epochs):
#         torch.cuda.empty_cache()
#         gc.collect()
#         student.train()
#         total_loss = 0

#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             with torch.no_grad():
#                 teacher_logits = teacher(images)

#             with amp.autocast(device_type="cuda"):
#                 student_logits = student(images)
#                 loss = criterion(student_logits, teacher_logits, labels)

            
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         scheduler.step(avg_train_loss)

#         print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

#         # Train Loss가 최소일 때 모델 저장
#         if avg_train_loss < best_train_loss:
#             best_train_loss = avg_train_loss
#             model_save_path = os.path.join(model_dir, "student_model.pth")
#             torch.save(student.state_dict(), model_save_path)
#             print(f"✅ Best Model Updated: {model_save_path} (Train Loss: {best_train_loss:.4f})")
#             patience_counter = 0
#         else:
#             patience_counter += 1

#         # Early Stopping 적용
#         if patience_counter >= early_stopping_patience:
#             print("Early Stopping")
#             break

import torch.nn.utils.prune as prune
def train_student_all(teacher, student, train_loader, epochs, lr, temperature, alpha, num_classes, patience):
    """
    Teacher 모델을 활용하여 Student 모델을 학습하는 함수 (Validation 없이 Train만 수행)

    Args:
        teacher (nn.Module): Teacher 모델
        student (nn.Module): MobileNet Student 모델
        train_loader (DataLoader): Train 데이터 로더
        epochs (int): 학습할 Epoch 수
        lr (float): Learning Rate
        temperature (float): 지식 증류 온도 값
        alpha (float): 지식 증류 가중치
        num_classes (int): 클래스 개수
        patience (int): Early Stopping 기준
    """
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "model_pth"
    feature_map_dir = "feature_maps"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(feature_map_dir, exist_ok=True)

    teacher = teacher.to(device).eval()

    # ✅ MobileNet Student의 출력층 조정
    student.classifier[3] = nn.Linear(1024, num_classes)  # MobileNet
    # student.fc = nn.Linear(1024, num_classes)  # ShuffleNetV2
    student = student.to(device).train()

    # from torch.nn.utils import prune

    # prune.ln_structured(student.features[0][0], name="weight", amount=0.5, n=2, dim=0) 
    # if hasattr(student.features[0][0], "weight_orig") and hasattr(student.features[0][0], "weight_mask"):
    #     prune.remove(student.features[0][0], "weight")
    #     print("✅ Pruning removed successfully!")
    # else:
    #     print("⚠️ Pruning was not applied properly, cannot remove.")
    # print("Pruned parameters:", list(student.features[0][0]._parameters.keys()))
    # print("Buffers:", list(student.features[0][0]._buffers.keys()))

    criterion = DistillationLoss(temperature, alpha)
    
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    scaler = amp.GradScaler()

    wandb.init(
    project="document_classification",
    name="MobileNetV3-Small",
    group="Student",
    config={
        "epochs": EPOCHES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__
    }
)

    # ✅ Hook을 사용하여 특징맵 저장
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    # ✅ 특정 레이어 선택 (MobileNetV3 기준: 첫 번째 Conv 레이어)
    hook_layer_name = "features.0"  # 첫 번째 Conv 레이어
    hook_handle = getattr(student.features, "0").register_forward_hook(get_activation(hook_layer_name))

    # ✅ Early Stopping 변수
    best_train_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = patience

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        student.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            with amp.autocast(device_type="cuda"):
                student_logits = student(images)
                loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # ✅ 특정 배치에서 특징맵 저장 (1개 샘플만 저장)
            if batch_idx == 0:  
                feature_map = activation[hook_layer_name][0]  # 첫 번째 샘플의 특징맵 저장
                num_features = feature_map.shape[0]  # 특징맵 개수

                # ✅ 특징맵 저장 디렉토리 설정
                save_path = os.path.join(feature_map_dir, f"epoch_{epoch+1}.npy")
                np.save(save_path, feature_map)
                print(f"📌 Feature Map saved at: {save_path}")

                # ✅ 특징맵 시각화 (16개만 선택)
                fig, axes = plt.subplots(4, 4, figsize=(20, 20))
                for i, ax in enumerate(axes.flat):
                    if i < num_features:
                        ax.imshow(feature_map[i], cmap="viridis")
                        ax.axis("off")
                plt.suptitle(f"Epoch {epoch+1} Feature Maps")
                plt.savefig(os.path.join(feature_map_dir, f"epoch_{epoch+1}.png"))
                plt.close()

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        #scheduler.step()
        
        wandb.log({"Student Train Loss": avg_train_loss, "Student Learning Rate": optimizer.param_groups[0]['lr'], "Student Epoch": epoch + 1})

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # ✅ Train Loss가 최소일 때 모델 저장
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            model_save_path = os.path.join(model_dir, "student_model.pth")

            torch.save(student.state_dict(), model_save_path)
            print(f"✅ Best Model Updated: {model_save_path} (Train Loss: {best_train_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # ✅ Early Stopping 적용
        if patience_counter >= early_stopping_patience:
            print("Early Stopping")
            break

    # ✅ Hook 제거
    hook_handle.remove()
    wandb.finish()
