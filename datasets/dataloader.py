from torchvision import transforms
from datasets.dataset import DocumentDataset, DocumentTestDataset

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from datasets.preprocess import collate_fn, collate_fn_test
from utils.config import IMAGE_SIZE


def get_test_data_loaders(data_dir, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = DocumentTestDataset(data_dir=f"{data_dir}/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_test)

    return test_loader


def get_data_loaders_with_kfold(data_dir, batch_size, random_seed, num_workers=4, n_splits=5, current_fold=0):
    """
    K-Fold Cross-Validation을 적용한 데이터 로더 생성
    Args:
        data_dir (str): 데이터 디렉토리 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로더 워커 수
        n_splits (int): K-Fold에서의 Fold 개수
        current_fold (int): 현재 Fold 번호 (0부터 시작)
    Returns:
        train_loader (DataLoader): K-Fold에서의 훈련 데이터 로더
        val_loader (DataLoader): K-Fold에서의 검증 데이터 로더
    """
    # Transform 설정
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DocumentDataset(data_dir=f"{data_dir}/train", transform=transform)
    labels = dataset.labels

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    train_indices, val_indices = list(skf.split(np.zeros(len(labels)), labels))[current_fold]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader


def get_train_data_loaders(data_dir, batch_size, num_workers=4):
    """
    validation 없이 train only
    Args:
        data_dir (str): 데이터 디렉토리 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로더 워커 수
    Returns:
        train_loader (DataLoader): 훈련 데이터 로더
    """
    # Transform 설정
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DocumentDataset(data_dir=f"{data_dir}/train", transform=transform)
    labels = dataset.labels

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn
    )

    return train_loader