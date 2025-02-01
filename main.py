import torch.nn as nn
from torchvision import models

from utils.gpu_utils import get_device, move_to_device
from utils.config import DATA_DIR, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES, LEARNING_RATE, TEMPERATURE, ALPHA, EPOCHES, K_FOLDS
from utils.seeds import set_seed

#from datasets.dataloader import get_data_loaders
from datasets.dataloader import get_data_loaders_with_kfold
from train import train_teacher, train_student

import torch
torch.cuda.empty_cache()  # GPU 캐시 메모리 비우기

def main():
    # 1. 시드 설정
    set_seed(RANDOM_SEED)

    # 2. Config 및 데이터 로더 설정
    data_dir = DATA_DIR
    num_classes = NUM_CLASSES
    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    temperature = TEMPERATURE
    alpha = ALPHA
    epochs = EPOCHES
    n_splits = K_FOLDS

    # 고정 validation 학습
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # train_loader, val_loader = get_data_loaders(data_dir, batch_size)

    # # 3. Teacher 모델 학습
    # teacher = models.efficientnet_b0(pretrained=True)
    # train_teacher(teacher, train_loader, val_loader, epochs, lr, num_classes)

    # # 4. Student 모델 학습 (지식 증류)
    # student = models.mobilenet_v3_small(pretrained=True)
    # train_student(teacher, student, train_loader, val_loader, epochs=epochs, lr=lr, temperature=temperature, alpha=alpha)
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


    for current_fold in range(n_splits):
        print(f"\n========== Fold {current_fold + 1}/{n_splits} ==========\n")
        
        # 3. K-Fold 데이터 로더 생성
        train_loader, val_loader = get_data_loaders_with_kfold(data_dir, batch_size, RANDOM_SEED, n_splits=n_splits, current_fold=current_fold)

        # 4. Teacher 모델 학습
        teacher = models.efficientnet_b0(pretrained=True)
        train_teacher(teacher, train_loader, val_loader, epochs, lr, num_classes, fold_idx=current_fold)

        # 5. Student 모델 학습 (지식 증류)
        student = models.mobilenet_v3_small(pretrained=True)
        train_student(teacher, student, train_loader, val_loader, epochs=epochs, lr=lr, temperature=temperature, alpha=alpha, num_classes=num_classes, fold_idx=current_fold)

        print(f"\n========== Fold {current_fold + 1} 완료 ==========\n")


if __name__ == "__main__":
    main()
