import timm
import torch


def load_EfficientNetV2(num_classes, type_size = "s", pretrained_path="", device="cpu"):
    if type_size == "s":
        # EfficientNetV2-S
        model = timm.create_model("tf_efficientnetv2_s", pretrained=True,
            num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)
        print("Loading EfficientNetV2_s model successfully!\n")
    elif type_size == "m":
        # EfficientNetV2-M
        model = timm.create_model("tf_efficientnetv2_m", pretrained=True,
            num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.3)
        print("Loading EfficientNetV2_m model successfully!\n")
    elif type_size == "l":
    # EfficientNetV2-L
        model = timm.create_model("tf_efficientnetv2_l", pretrained=True, 
            num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.4)
    print("Loading EfficientNetV2_l model successfully!\n")
    if pretrained_path != "":
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")
    return model