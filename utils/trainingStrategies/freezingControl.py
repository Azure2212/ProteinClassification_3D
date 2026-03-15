def freeze_backbone(model, keep_head=True):
    for p in model.parameters():
        p.requires_grad = False

    if keep_head:
        for name, module in model.named_children():
            if name in ["head", "fc", "classifier"]:
                for p in module.parameters():
                    p.requires_grad = True


def unfreeze_last_n_stages(model, n=2):

    # ResNet / RegNet
    if hasattr(model, "layer1"):
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    # EfficientNetV2 (timm-style)
    elif hasattr(model, "blocks"):
        stages = list(model.blocks)
        for p in model.conv_head.parameters():
            p.requires_grad = True
        for p in model.bn2.parameters():
            p.requires_grad = True

    # Torchvision ConvNeXt (uses features)
    elif hasattr(model, "features"):
        # Keep only stage blocks (skip stem + downsample layers)
        stages = [
            model.features[i]
            for i in range(len(model.features))
            if i % 2 == 1   # 1,3,5,7 are actual stages
        ]

    # RegNetY16GF (torchvision)
    elif hasattr(model, "s1") and hasattr(model, "s4"):
        stages = [model.s1, model.s2, model.s3, model.s4]
    
    # Swin / MaxViT
    elif hasattr(model, "layers"):
        stages = list(model.layers)

    # CoAtNet or others
    elif hasattr(model, "stages"):
        stages = list(model.stages)

    else:
        raise ValueError(
            f"Unsupported architecture structure: {model.__class__.__name__}"
        )

    # Unfreeze last n stages
    for s in stages[-n:]:
        for p in s.parameters():
            p.requires_grad = True

    # Unfreeze final norm if exists
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True
    

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True