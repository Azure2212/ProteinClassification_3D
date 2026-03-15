import torch

# Recommended decay per model:
#   Swin / MaxViT:  decay = 0.85
#   CoAtNet:        decay = 0.90  (CNN stages benefit from more LR)
 
# ── Always exclude bias and LayerNorm from weight_decay ──────────────

def specificOptimizerPerModel(modelName, model, learning_rate):

    # 🔵 Transformer models
    if modelName in ["SwinV2B", "MaxViT", "CoAtNet"]:

        no_wd_keys = {
            "bias","LayerNorm.weight","norm.weight","norm.bias",
            "pos_embed","cls_token","relative_position_bias_table"
        }

        wd_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(k in n for k in no_wd_keys)
        ]

        nwd_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in no_wd_keys)
        ]

        param_groups = [
            {"params": wd_params,  "weight_decay": 0.05},
            {"params": nwd_params, "weight_decay": 0.0}
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    # 🟢 EfficientNetV2
    elif modelName in ["EfficientNetV2", "Resnet50", "ConvNeXt"]:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-2
        )

    # 🟠 CNN models (SGD)
    elif modelName in ["RegNetY16GF"]:
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4
        )

    else:
        raise ValueError(f"Model {modelName} is not supported.")