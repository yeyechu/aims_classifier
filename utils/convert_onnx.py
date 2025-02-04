import torch
import onnx
import os
from torchvision import models

# 변환할 가중치 파일 경로
ROOT = "/data/ephemeral/home/aims_image_classifier/model_pth"
MODEL_PATH = ROOT + "/student_model.pth"
ONNX_PATH = ROOT + "/student_model.onnx"

# Student 모델 정의
def load_student_model(num_classes=6):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    return model

# ✅ PyTorch 모델 로드
num_classes = 6
model = load_student_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ✅ ONNX 변환을 위한 더미 입력 생성 (배치 크기 1, 3채널, 224x224)
dummy_input = torch.randn(1, 3, 512, 512)

# ✅ ONNX 변환 실행
torch.onnx.export(
    model, dummy_input, ONNX_PATH,
    export_params=True, opset_version=11,  # 최신 ONNX 버전 사용
    do_constant_folding=True,  # 최적화 적용
    input_names=["input"], output_names=["output"]
)

print(f"✅ ONNX 모델 저장 완료: {ONNX_PATH}")
