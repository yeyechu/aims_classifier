import torch.nn as nn


class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, target):
        # Teacher와 Student의 Soft Logits
        student_soft = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.temperature, dim=1)

        # Knowledge Distillation 손실 (KL Divergence)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Hard Label 손실 (CrossEntropy)
        cross_entropy_loss = nn.functional.cross_entropy(student_logits, target)

        # 최종 손실
        return self.alpha * distillation_loss + (1 - self.alpha) * cross_entropy_loss