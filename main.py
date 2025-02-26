import os
import torch.nn as nn
from torchvision import models

from utils.config import DATA_DIR, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES, LEARNING_RATE, TEMPERATURE, ALPHA, EPOCHES, K_FOLDS, PATIENCE, NUM_WORKERS
from utils.seeds import set_seed

#from datasets.dataloader import get_data_loaders
from datasets.dataloader import get_data_loaders_with_kfold, get_train_data_loaders

from models.student_models import CustomMobileNet
from train import train_teacher, train_student, train_student_with_fd, train_student_with_va, train_teacher_all, train_student_all

import torch
torch.cuda.empty_cache()  # GPU 캐시 메모리 비우기

#import wandb


def main():
    # 1. 시드 설정
    set_seed(RANDOM_SEED)

    # 2. Config 및 데이터 로더 설정
    data_dir = DATA_DIR
    num_classes = NUM_CLASSES
    num_workers = NUM_WORKERS
    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    temperature = TEMPERATURE
    alpha = ALPHA
    epochs = EPOCHES
    n_splits = K_FOLDS
    patience = PATIENCE

    model_dir = "model_pth"
    # 3. 데이터 로더
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    train_loader = get_train_data_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    # 4. Teacher 모델 학습
    teacher = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    train_teacher_all(teacher, train_loader, epochs, lr, num_classes, patience)

    teacher_model_path = os.path.join(model_dir, f"teacher_model.pth")
    
    teacher = models.efficientnet_b0(weights=None)
    teacher.classifier[1] = nn.Linear(1280, num_classes)
    teacher.load_state_dict(torch.load(teacher_model_path))
    teacher.eval().cuda()

    # 5. Student 모델 학습 (지식 증류)
    student = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    #student = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    train_student_all(teacher, student, train_loader, epochs, lr, temperature, alpha, num_classes, patience)
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    # for current_fold in range(n_splits):
    #     print(f"\n========== Fold {current_fold + 1}/{n_splits} ==========\n")
        
    #     # 3. K-Fold 데이터 로더 생성
    #     train_loader, val_loader = get_data_loaders_with_kfold(data_dir, batch_size, RANDOM_SEED, n_splits=n_splits, current_fold=current_fold)

    #     # 4. Teacher 모델 학습
    #     #teacher = CustomEfficientNet(num_classes=num_classes)
    #     #teacher = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    #     # train_teacher(teacher, train_loader, val_loader, epochs, lr, num_classes, fold_idx=current_fold, patience=patience)

    #     # 4-1. Teacher 모델 학습없이 불러오기
    #     teacher_model_path = os.path.join(model_dir, f"teacher_model_fold{current_fold}.pth")  # Fold별 Teacher 가중치
        
    #     teacher = models.efficientnet_b0(weights=None)
    #     teacher.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    #     teacher.load_state_dict(torch.load(teacher_model_path))
        
    #     #teacher_base = models.efficientnet_b0(weights=None)
    #     #teacher_base.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    #     #teacher_base.load_state_dict(torch.load(teacher_model_path), strict=False)
    #     #teacher = FeatureExtractorWrapper(teacher_base)
        
    #     teacher.eval().cuda()

    #     # 5. Student 모델 학습 (지식 증류)
    #     student = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    #     #train_student(teacher, student, train_loader, val_loader, epochs=epochs, lr=lr, temperature=temperature, alpha=alpha, num_classes=num_classes, fold_idx=current_fold, patience=patience)
    #     train_student_with_va(teacher, student, train_loader, val_loader, epochs=epochs, lr=lr, temperature=temperature, alpha=alpha, num_classes=num_classes, fold_idx=current_fold, patience=patience)
        
    #     #student = CustomMobileNet(num_classes=num_classes)
    #     #train_student_with_fd(teacher, student, train_loader, val_loader, epochs=epochs, lr=lr, temperature=temperature, alpha=alpha, num_classes=num_classes, fold_idx=current_fold, patience=patience)
        

    #     print(f"\n========== Fold {current_fold + 1} 완료 ==========\n")



if __name__ == "__main__":
    main()
