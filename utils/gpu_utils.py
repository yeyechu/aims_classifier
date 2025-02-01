import torch

def get_device():
    """
    GPU 사용 가능 여부를 확인하고 장치를 반환합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    return device

def move_to_device(data, device):
    """
    데이터를 GPU로 전송합니다.
    """
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device)
