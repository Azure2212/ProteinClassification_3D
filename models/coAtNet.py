import timm
import torch

def load_CoAtNet(name, num_classes, pretrained_path="", device="cpu"):
    if name == "CoAtNet_2":
        model = timm.create_model("coatnet_2_rw_224", pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.3)
        print("Loading CoAtNet_2 model successfully!\n")
    elif name == "CoAtNet_3":
        model = timm.create_model("coatnet_3_rw_224", pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.3)
        print("Loading CoAtNet_3 model successfully!\n")
    elif name == "CoAtNet_4":
        model = timm.create_model("coatnet_4_rw_224", pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.3)
        print("Loading CoAtNet_4 model successfully!\n")
    else:
        raise ValueError(f"Unsupported model name: {name}. Expected 'CoAtNet_2'.")
    if pretrained_path != "":
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")
    return model