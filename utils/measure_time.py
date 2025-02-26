import torch
import time
from torchvision import models, transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.mobilenet_v3_small(weights=None)
# model.classifier[3] = torch.nn.Linear(1024, 6)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 6)

model.to(device)
model.eval()

# ✅ 실제 이미지 로드 (inference.py와 동일한 데이터셋 사용)
image_path = "/data/ephemeral/home/aims_image_classifier/data/test/image_054.png"
size = 224

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])
image = Image.open(image_path).convert("RGB")
x = transform(image).unsqueeze(0).to(device)

# ✅ Warm-up 실행
for _ in range(10):
    _ = model(x)
torch.cuda.synchronize()

# ✅ 실시간 추론 속도 측정
times = []
num_tests = 10
with torch.no_grad():
    for _ in range(num_tests):
        start_time = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        times.append((time.time() - start_time) * 1000)  # ms 단위

avg_latency = sum(times) / len(times)
print(f"🚀 Web 실시간 추론 속도: {avg_latency:.3f} ms (batch_size=1)")
