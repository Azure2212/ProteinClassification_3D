import torch
import torch.nn as nn
import torchvision.models as models

def load_Resnet(name, num_classes, pretrained_path="", device="cpu"):
    if name == "Resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        print("Loading ResNet50 model successfully!\n")
    elif name == "Resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        print("Loading ResNet101 model successfully!\n")
    elif name == "Resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        print("Loading ResNet152 model successfully!\n")
    else:
        raise ValueError(f"Unsupported model name: {name}. Expected 'Resnet50'.")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if pretrained_path != "":
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")
    return model