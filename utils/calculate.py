import os
import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from config import IMAGE_SIZE, NUM_CLASSES

# ✅ 파인 튜닝된 EfficientNet-B0 모델 불러오기
model_dir = "model_pth"
teacher_model_path = os.path.join(model_dir, f"teacher_model.pth")
student_model_path = os.path.join(model_dir, f"student_model.pth")

num_classes = NUM_CLASSES
image_size = IMAGE_SIZE


# model = models.efficientnet_b0(weights=None)
# model.classifier[1] = torch.nn.Linear(1280, num_classes)
# model.load_state_dict(torch.load(teacher_model_path, map_location="cuda"))


model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(1024, num_classes)
model.load_state_dict(torch.load(student_model_path, map_location="cuda"))



model.eval().cuda()

# ✅ 입력 데이터 샘플 (배치 1개, 3채널, 512x512 해상도)
input_tensor = torch.randn(1, 3, image_size, image_size).cuda()

# ✅ FLOPs 및 파라미터 개수 계산
flops = FlopCountAnalysis(model, input_tensor)
params = parameter_count_table(model)

# ✅ 결과 출력
#print(f"EffcientNet-B0({image_size}) Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print(f"MobileNetV3-Small({image_size}) Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print(params) 
