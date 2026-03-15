import timm
import torch


def load_EfficientNetV2(num_classes, type_size = "m", pretrained_path="", device="cpu"):
    if type_size == "s":
        # EfficientNetV2-S
        model = timm.create_model("tf_efficientnetv2_s", pretrained=True,
            num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)
    elif type_size == "m":
        # EfficientNetV2-M
        model = timm.create_model("tf_efficientnetv2_m", pretrained=True,
            num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.3)

    if pretrained_path != "":
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")
    return model