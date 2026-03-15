import math
import torch
 
def cosine_warmup_schedule(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
 
# For EfficientNet / RegNet, OneCycleLR is also a strong alternative:
# torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=3e-4, total_steps=total_steps)
