import timm
import torch

def load_VIT_SizeT(num_classes, pretrained_path="", device="cpu"):
    model = timm.create_model("maxvit_tiny_tf_224", pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)

    if pretrained_path != "":
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")
    return model