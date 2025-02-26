# 데이터 전처리 (이미지 및 OCR)
import cv2
import torch

def collate_fn(batch):
    images, labels = zip(*batch)

    # 모든 이미지의 최대 크기 찾기
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=1.0)  # 흰색 패딩
        padded_images.append(padded_img)

    padded_images = torch.stack(padded_images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_images, labels


def collate_fn_test(batch):
    """
    DataLoader에서 배치 데이터를 (이미지, 파일명)으로 변환하는 collate_fn
    """
    images, file_names = zip(*batch)

    # 가장 큰 높이, 너비를 찾아서 패딩 적용
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    padded_images = []
    for img in images:
        pad_h = max_height - img.shape[1]
        pad_w = max_width - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)  # 오른쪽, 아래쪽에 패딩 추가
        padded_images.append(padded_img)

    images = torch.stack(padded_images, dim=0)  # 동일한 크기로 정렬 후 스택

    return images, list(file_names)


def resize_with_padding(image, target_size):
    """
    비율을 유지하면서 세로 크기에 맞추고 가로 크기를 패딩하여 리사이즈
    Args:
        image (numpy array): 입력 이미지
        target_size (int): 리사이즈할 이미지 크기 (정사각형 크기)
    Returns:
        torch.Tensor: 전처리된 이미지 텐서
    """
    h, w, c = image.shape
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h))

    # 패딩 추가
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)

    padded_image = cv2.copyMakeBorder(
        resized_image, 
        pad_top, pad_bottom, pad_left, pad_right, 
        borderType=cv2.BORDER_CONSTANT, 
        value=(255, 255, 255)  # 흰색 패딩
    )

    # 이미지 정규화 및 텐서 변환
    padded_image = padded_image / 255.0
    padded_image = torch.tensor(padded_image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)

    return padded_image
