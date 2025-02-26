import torch
import wandb
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os

from loss import DistillationLoss
import torch.amp as amp


from datasets.dataloader import get_train_data_loaders
from utils.config import DATA_DIR, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES, LEARNING_RATE, TEMPERATURE, ALPHA, EPOCHES, K_FOLDS, PATIENCE, NUM_WORKERS
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

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, 6)

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


# ✅ 옵티마이저 리스트
optimizers = {
    "SGD": optim.SGD(model.parameters(), lr=lr),
    "SGD_Momentum": optim.SGD(model.parameters(), lr=lr, momentum=0.9),
    "Adam": optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4),
    "AdamW": optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4),
    "RMSprop": optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, momentum=0.9)
}

results = {}

for name, optimizer in optimizers.items():
    print(f"\n🚀 Testing {name} Optimizer")
    wandb.init(project="optimizer_experiment", name=name, group="Optimizer_Test")

    loss_values = []

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

        #wandb.log({"Teacher Loss": avg_loss})
        wandb.log({"Student Loss": avg_loss})
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

    wandb.finish()
    results[name] = loss_values
