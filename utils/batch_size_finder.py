import torch
from torchvision import models

def find_max_batch_size(model, image_size=(3, 512, 512), max_mem_ratio=0.9):
    """
    GPU 메모리를 초과하지 않는 최대 batch_size를 탐색하는 함수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = 1
    max_batch_size = 1

    while True:
        try:
            dummy_input = torch.randn(batch_size, *image_size).to(device)
            _ = model(dummy_input)  # 추론 테스트
            torch.cuda.synchronize()

            # GPU 메모리 사용량 확인
            allocated_mem = torch.cuda.memory_allocated(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory

            # 최대 메모리 대비 사용률 계산
            if allocated_mem / total_mem > max_mem_ratio:
                break  # 메모리 초과 가능성이 있으면 중단

            max_batch_size = batch_size
            batch_size += 1  # 배치 크기 증가

        except RuntimeError:  # OOM 발생하면 중단
            break

    return max_batch_size


# model = models.mobilenet_v3_small(weights=None)
# model.classifier[3] = torch.nn.Linear(1024, 6)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, 6)


optimal_batch_size = find_max_batch_size(model)
print(f"🚀 최적의 batch_size: {optimal_batch_size}")
