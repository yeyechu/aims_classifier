import os
import cv2
import torch
from PIL import Image

from torch.utils.data import Dataset
from utils.config import IMAGE_SIZE
from datasets.preprocess import resize_with_padding


class DocumentDataset(Dataset):
    """
    PyTorch Dataset 클래스 구현
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 데이터 디렉토리 경로
            transform (callable, optional): 데이터 전처리 함수
        """
        self.data_dir = data_dir
        self.transform = transform

        # 클래스 디렉토리 가져오기
        self.class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.class_dirs.sort()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_dirs)}

        # 이미지 파일과 라벨 수집
        self.image_paths = []
        self.labels = []
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith((".jpg", ".png")):
                    self.image_paths.append(os.path.join(class_path, file_name))
                    self.labels.append(class_idx)


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 기본적으로 BGR 형식이므로 RGB로 변환
        image = Image.fromarray(image) 
        
        if self.transform:
            image = self.transform(image)
        else:
            image = resize_with_padding(image, IMAGE_SIZE)
            

        return image, torch.tensor(label, dtype=torch.long)


class DocumentTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        file_name = os.path.basename(img_path)
        return image, file_name


