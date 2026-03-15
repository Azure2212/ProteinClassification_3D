import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode

def LoadData(dataset_folder, class_names, isDebug=0, showSize = True):
    images = []
    labels = []
    allProteins = os.listdir(dataset_folder)
    allProteins = sorted(allProteins)
    for k, v in class_names.items():
        protein_folder = os.path.join(dataset_folder, v)
        allImages = os.listdir(protein_folder)
        allImages = sorted(allImages)
 
        for image in allImages:
            image_path = os.path.join(protein_folder, image)
            threeChannelImage = cv2.imread(image_path)[:,:,::-1]
            images.append(threeChannelImage)
            labels.append(k)
            if isDebug == 1:
                break
    if showSize:
        print(f"Total images: {len(images)} --- Total labels: {len(labels)}")
    return images, labels

#============ Data Augmentation ============#

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def train_tf(image_size):
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(360, fill=0),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x + 0.02 * torch.randn_like(x)).clamp(0.0, 1.0)),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        #transforms.RandomErasing(p=0.2),
    ])

def val_tf(image_size):
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
def test_tf(image_size):
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


class PBD42Dataset(Dataset):
    def __init__(self, images, labels, image_size, type_transform="train"):
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.transform = type_transform
        self.train_tf = train_tf(image_size)
        self.val_tf = val_tf(image_size)
        self.test_tf = test_tf(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        #img = (img * 255).astype(np.uint8)
        #img = Image.fromarray(img) 
        
        if self.transform == "train":
            img = self.train_tf(img)
        elif self.transform == "val":
            img = self.val_tf(img)
        elif self.transform == "test":
            img = self.test_tf(img)
        label = self.labels[idx]
        return img, label
    
def real_protein_testset(test_image_path, class_names):
    images_per_class = []
    labels_per_class = []
    allStructureInTest = {0: "HYDROLASE", 1: "LIGASE", 2: "METAL_BINDING_PROTEIN", 3: "OXIDOREDUCTASE"}
    for _, v in allStructureInTest.items():
        
        structures = os.path.join(test_image_path, v)
        #print(structures)
        proteins = os.listdir(structures)
        for protein in proteins:
            path = os.path.join(structures, protein)
            label_to_find = protein.upper()
            label = next((k for k, v in class_names.items() if v == label_to_find), -1)
            image_paths = os.listdir(path)
            for image_path in image_paths:
                images_per_class.append(os.path.join(path,image_path))
                labels_per_class.append(label)
    return images_per_class, labels_per_class