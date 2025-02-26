import os
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import models
from loss import DistillationLoss
from datasets.dataloader import get_train_data_loaders
from utils.config import DATA_DIR, BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, TEMPERATURE, ALPHA, EPOCHES, K_FOLDS, PATIENCE, NUM_WORKERS
import torch.amp as amp

data_dir = DATA_DIR
num_classes = NUM_CLASSES
num_workers = NUM_WORKERS
batch_size = BATCH_SIZE
lr = LEARNING_RATE
temperature = TEMPERATURE
alpha = ALPHA
epochs = EPOCHES
patience = PATIENCE

model_dir = "model_pth"
train_loader = get_train_data_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# model.classifier[1] = torch.nn.Linear(1280, 6)

teacher_model_path = os.path.join(model_dir, f"teacher_model.pth")
    
teacher = models.efficientnet_b0(weights=None)
teacher.classifier[1] = nn.Linear(1280, num_classes)

teacher.load_state_dict(torch.load(teacher_model_path))
teacher.eval().cuda()
teacher = teacher.to(device).eval()

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(1024, 6)  # Custom class 수 설정

model.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = DistillationLoss(temperature, alpha)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# ✅ 실험할 스케줄러 리스트
schedulers = {
    "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
    "MultiStepLR": optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5),
    "ExponentialLR": optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6),
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
}

# ✅ 스케줄러 별 실험 수행
results = {}

for name, scheduler in schedulers.items():
    print(f"\n🚀 Testing {name} Scheduler")
    wandb.init(project="scheduler_experiment", name=name, group="LR_Scheduler_Test")

    loss_values = []
    lr_values = []

    for epoch in range(50):  # 50 Epoch 실험
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            with amp.autocast(device_type="cuda"):
                student_logits = model(images)
                loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            outputs = model(images)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)
        lr_values.append(scheduler.get_last_lr()[0])

        # 스케줄러 업데이트
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)  # ReduceLROnPlateau은 loss 기반 업데이트
        else:
            scheduler.step()

        #wandb.log({"Teacher Loss": avg_loss, "Teacher Learning Rate": lr_values[-1]})
        wandb.log({"Student Loss": avg_loss, "Student Learning Rate": lr_values[-1]})
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={lr_values[-1]:.6f}")

    wandb.finish()
    results[name] = {"loss": loss_values, "lr": lr_values}

# ✅ 결과 시각화 (Loss & Learning Rate 비교)
plt.figure(figsize=(12, 5))

# Loss 곡선 비교
plt.subplot(1, 2, 1)
for name, result in results.items():
    plt.plot(result["loss"], label=name)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison")
plt.legend()

# Learning Rate 변화 곡선 비교
plt.subplot(1, 2, 2)
for name, result in results.items():
    plt.plot(result["lr"], label=name)
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()

plt.show()
