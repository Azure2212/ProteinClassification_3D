############# Importing System Libraries #############
import sys
import os
import time 

root_project_dir = os.getcwd().split("/")[:4]
root_project_dir = "/".join(root_project_dir)

############# Importing support Libraries #############
import json
import random
import numpy as np
import torch
import argparse

############# Importing datasets classes #############

sys.path.append(f"{root_project_dir}/utils/datasets")
from torch.utils.data import DataLoader
from pdb_ds import LoadData, PBD42Dataset, real_protein_testset
from get_classes import get_classes
############# Importing models #############
sys.path.append(f"{root_project_dir}/models")
# from resnet import load_resnet50
# from convnext import load_ConvNeXt
# from coAtNet2 import load_CoAtNet2
# from efficientNetV2 import load_efficientNetV2
# from maxViT import load_VIT_SizeT
# from regNetY16GF import load_regNetY16GF
# from swinV2B import load_swinV2B   
from models import (
    load_Resnet,
    load_ConvNeXt,
    load_CoAtNet,
    load_EfficientNetV2,
    load_VIT_SizeT,
    load_RegNetY16GF,
    load_SwinV2B,
)

############# Importing trainers #############
sys.path.append(f"{root_project_dir}/trainer")
from PDB42_Trainer import PDB42_Trainer

############# Importing evaluations #############
sys.path.append(f"{root_project_dir}/evaluations")
from evaluation_pdb import realTest_cm

############# Setting Random Seeds(Reproducibility) #############
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Random seed set to {SEED}")
#/data2/atran16/Anaconda_ForTrain/bin/python pdb_run.py --model EfficientNetV2 --image_size 384
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model EfficientNetV2 --image_size 384
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model CoAtNet --image_size 224
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model MaxViT --image_size 224
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model SwinV2B --image_size 256
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model RegNetY16GF --image_size 224
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model ConvNeXt --image_size 224
#/data2/atran16/Anaconda_ForTrain/bin/python /data2/atran16/ProteinClassification_AnhTuanTran/pdb_run.py --model Resnet50 --image_size 224
#screen -S mysessionTuan
#Ctrl + A, then D -> leave the screen session running in the background
#screen -r mysessionTuan
#Optional: screen -ls -> list all screen sessions
#screen -S mysessionTuan -X quit -> kill the screen session when you are done
#=============== Configurations ================#



parser = argparse.ArgumentParser(description="Prepare data and run training for Protein Classification")

# Paths (now optional because they have defaults)
parser.add_argument("--train_protein_path", type=str, 
                    default="/data/atran16/ProteinClassification_3D/3D_PDB_Dataset/TrainProteinPNG600")

parser.add_argument("--valid_protein_path", type=str,
                    default="/data/atran16/ProteinClassification_3D/3D_PDB_Dataset/ValidProteinPNG24")

parser.add_argument("--test_image_path", type=str,
                    default="/data/atran16/ProteinClassification_3D/3D_PDB_Dataset/testingDataFromProfessorSu")

parser.add_argument("--full_rs_dir", type=str,
                    default="/data/atran16/ProteinClassification_3D/trained_results/03142026_train/EfficientNetV2_L")

# Data parameters
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--image_each_proteins", nargs=2, type=int, default=[600, 60])
parser.add_argument("--n_channels", type=int, default=3)
parser.add_argument("--n_classes", type=int, default=127)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--isDebug", type=int, default=0)

# Model
parser.add_argument("--model", type=str, default="Resnet50") #"EfficientNetV2", "Resnet50", "ConvNeXt", "RegNetY16GF", "SwinV2B", "MaxViT", "CoAtNet"
parser.add_argument("--pretrained_path", type=str, default="")

# Optimization
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--min_lr", type=float, default=1e-7)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--optimizer_chose", type=str, default="RAdam")
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")

# Training
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--earlyStopping", type=int, default=1000)
parser.add_argument("--max_epoch_num", type=int, default=60)
parser.add_argument("--plateau_patience", type=int, default=10)
parser.add_argument("--steplr", type=int, default=50)
parser.add_argument("--log_step", type=int, default=1)

# Logging
parser.add_argument("--project_name", type=str, default="ProteinClassification")
parser.add_argument("--wandb_api_key", type=str, default="Qbo3kwrtRENVIG8hqrfSVpYn5dc_mvxp0fRvc4DE")
parser.add_argument("--name_run_wandb", type=str, default="Tuan_dep_trai_ne")

args = parser.parse_args()
configs = vars(args)
configs['rs_dir'] = os.path.join(configs['full_rs_dir'], "PDBRSTuan.pt")
configs['tracking_csv'] = os.path.join(configs['full_rs_dir'], "trainingTracking.csv")
configs["image_size"] = (configs["image_size"], configs["image_size"])
print("Configurations:")
for key, value in configs.items():
    print(f"{key}: {value}")
    
save_path = os.path.join(configs['full_rs_dir'], "configs.json")

with open(save_path, "w") as f:
    json.dump(configs, f, indent=4)

class_names = get_classes(configs["train_protein_path"])
topk=(1,3,5,10,20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\ntop k: {topk}")
print(f"The device is being used is: {device}\n")

#=============== Loading Model ================#
if "Resnet" in configs["model"]:
    model = load_Resnet(name = configs["model"], num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
elif configs["model"] == "ConvNeXt":
    model = load_ConvNeXt(num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
    print("Loading ConvNeXt model successfully!\n")
elif "CoAtNet" in configs["model"]:
    model = load_CoAtNet(name = configs["model"], num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
    print("Loading CoAtNet2 model successfully!\n")
elif "EfficientNetV2" in configs["model"]:
    model = load_EfficientNetV2(name = configs["model"], num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
elif configs["model"] == "MaxViT":
    model = load_VIT_SizeT(num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
    print("Loading MaxViT_SizeT model successfully!\n")
elif configs["model"] == "RegNetY16GF":
    model = load_RegNetY16GF(num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
    print("Loading RegNetY16GF model successfully!\n")
elif configs["model"] == "SwinV2B":
    model = load_SwinV2B(num_classes = configs["n_classes"], pretrained_path=configs["pretrained_path"], device=device)
    model = model.to(device)
    print("Loading SwinV2B model successfully!\n")
else:
    raise ValueError(f"Unsupported model type: {configs['model']}")

#=============== Loading Dataset ================#
start_time = time.time()
train_image, train_label = LoadData(dataset_folder = configs["train_protein_path"], class_names=class_names, isDebug=configs['isDebug'])
val_image, val_label = LoadData(dataset_folder = configs["valid_protein_path"], class_names=class_names, isDebug=configs['isDebug'])

print(f"number images in train data: {len(train_image)}({100*len(train_image)//(len(train_image)+len(val_image))}%)")
print(f"number images in val data: {len(val_image)}({100*len(val_image)//(len(train_image)+len(val_image))}%)\n")

train_data = PBD42Dataset(train_image, train_label, image_size=configs["image_size"], type_transform="train")
val_data   = PBD42Dataset(val_image, val_label, image_size=configs["image_size"], type_transform="val")

train_loader = DataLoader(train_data, batch_size = configs["batch_size"], shuffle=True)
val_loader = DataLoader(val_data , batch_size = configs["batch_size"], shuffle=False)

end_time = time.time() 
print(f"Total time For Loading and Preparing dataset: {end_time - start_time:.2f} seconds")
#=============== Setup Trainer ================#

trainer = PDB42_Trainer(
    model=model,
    device=device,
    configs=configs,
    class_names=class_names,
    topk=topk
)

#=============== Training start ================#
trainer.run(
    epochs=configs['max_epoch_num'],
    train_loader=train_loader,
    val_loader=val_loader,
    log_step=configs['log_step']
)

#=============== Real classification result ================#
images_per_class, labels_per_class = real_protein_testset(configs["test_image_path"], class_names)
for k in topk:
    realTest_cm(
        image_size=configs["image_size"],
        class_names=class_names,
        checkpoint_path=configs["rs_dir"],
        device=device,
        model=model,
        path2save=configs["full_rs_dir"],
        images_per_class=images_per_class,
        labels_per_class=labels_per_class,
        top_k = k,
        saveStatisticsReport=True
    )