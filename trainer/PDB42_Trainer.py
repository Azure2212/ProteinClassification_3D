import sys
import os
import torch
import torch.nn as nn
import tqdm
import csv

root_project_dir = os.getcwd().split("/")[:4]
root_project_dir = "/".join(root_project_dir)

sys.path.append(os.path.join(root_project_dir, "evaluations"))
sys.path.append(os.path.join(root_project_dir, "utils/trainingStrategies"))

from evaluation_pdb import line_chart_k_acc, line_chart
from specificOptimizerPerModel  import specificOptimizerPerModel 
from specificLRSchedulerPerModel import cosine_warmup_schedule
from freezingControl import freeze_backbone, unfreeze_last_n_stages, unfreeze_all

class PDB42_Trainer:
    def __init__(self, model, device,
                 configs, class_names=None, topk=(1,3,5,10,20)):

        self.model = model.to(device)
        self.device = device     
        self.topk = topk
        self.configs = configs
        self.class_names = class_names

        # Tracking
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.lr_list = []

        self.best_val_acc = 0.0
        self.best_val_loss = 0.0
        self.best_train_acc = 0.0
        self.best_train_loss = 0.0

        #Training configuration
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = specificOptimizerPerModel(
            modelName = configs["model"],
            model = self.model,
            learning_rate = configs["lr"]
        )
        # Learning Rate Scheduler
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=configs["max_epoch_num"],
        #     eta_min=1e-6
        # )
        total_steps = configs["max_epoch_num"]
        warmup_steps = int(0.1 * total_steps) 

        self.lr_scheduler = cosine_warmup_schedule(
            optimizer=self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )

        #Creating Tracking

        header = ["epoch", "train_acc", "train_loss", "val_acc", "val_loss",
            "topk1train_acc", "topk3train_acc", "topk5train_acc", "topk10train_acc", "topk20train_acc",
            "topk1val_acc", "topk3val_acc", "topk5val_acc", "topk10val_acc", "topk20val_acc", "learning_rate"]
        
        with open(configs["tracking_csv"], mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print("TrackingTraining CSV is created!")
        
    def apply_finetune_strategy(self, epoch):

        if epoch == 1:
            print("Stage 1: Head only")
            freeze_backbone(self.model)
            # total_params = 0
            # trainable_params = 0

            # for name, param in self.model.named_parameters():
            #     status = "Trainable" if param.requires_grad else "Frozen"
            #     print(f"{name:60} | {status}")

            #     total_params += param.numel()
            #     if param.requires_grad:
            #         trainable_params += param.numel()

        elif epoch == 6:
            print("Stage 2: Unfreeze last 2 stages")
            unfreeze_last_n_stages(self.model, n=2)
            # total_params = 0
            # trainable_params = 0

            # for name, param in self.model.named_parameters():
            #     status = "Trainable" if param.requires_grad else "Frozen"
            #     print(f"{name:60} | {status}")

            #     total_params += param.numel()
            #     if param.requires_grad:
            #         trainable_params += param.numel()

        elif epoch == 11:
            print("Stage 3: Full fine-tuning")
            unfreeze_all(self.model)
            # total_params = 0
            # trainable_params = 0

            # for name, param in self.model.named_parameters():
            #     status = "Trainable" if param.requires_grad else "Frozen"
            #     print(f"{name:60} | {status}")

            #     total_params += param.numel()
            #     if param.requires_grad:
            #         trainable_params += param.numel()

    def train_one_epoch(self, train_loader, epoch):

        self.model.train()
        total_loss = 0.0
        total_examples = 0
        total_correct = {f"{k}": 0 for k in self.topk}

        for images, labels in tqdm.tqdm(
            train_loader,
            total=len(train_loader),
            leave=True,
            colour="blue",
            desc=f"Epoch {epoch}"
        ):

            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            # ---- Top-K computation ----
            max_k = max(self.topk)
            _, pred = torch.topk(logits, k=max_k, dim=1)  # shape: [B, max_k]

            for k in self.topk:
                correct_k = (pred[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
                total_correct[f"{k}"] += correct_k

        avg_loss = total_loss / total_examples
        avg_acc = {k: total_correct[k] / total_examples for k in total_correct}
        
        return avg_loss, avg_acc

    def evaluate(self, val_loader, epoch):

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        total_correct = {f"{k}": 0 for k in self.topk}

        with torch.no_grad():
            for images, labels in tqdm.tqdm(
                val_loader,
                total=len(val_loader),
                leave=True,
                colour="green",
                desc=f"Epoch {epoch}"
            ):

                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                # ---- Top-K computation ----
                max_k = max(self.topk)
                _, pred = torch.topk(logits, k=max_k, dim=1)  # shape: [B, max_k]

                for k in self.topk:
                    correct_k = (pred[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
                    total_correct[f"{k}"] += correct_k


        avg_loss = total_loss / total_examples
        avg_acc = {k: total_correct[k] / total_examples for k in total_correct}

        return avg_loss, avg_acc


    def run(self, epochs, train_loader, val_loader, log_step=0):

        early_stop_flag = 0

        for epoch in range(1, epochs + 1):
            
            self.apply_finetune_strategy(epoch)

            train_loss, train_acc_dict = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc_dict = self.evaluate(val_loader, epoch)

            train_acc = train_acc_dict["1"]
            val_acc = val_acc_dict["1"]

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)

            # Clamp minimum LR
            for param_group in self.optimizer.param_groups:
                if param_group["lr"] < self.configs["min_lr"]:
                    param_group["lr"] = self.configs["min_lr"]

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_list.append(current_lr)

            print(
                f"Epoch: {epoch:02d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | "
                f"LR: {current_lr:.7f}"
            )

            print("Train Top-k, ", 
                  " || ".join([f"top{k}:{v*100:.2f}%" for k,v in train_acc_dict.items()]))

            print("Val Top-k, ", 
                  " || ".join([f"top{k}:{v*100:.2f}%" for k,v in val_acc_dict.items()]))

            self.lr_scheduler.step()

            # Save best model (based on Top1 validation)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_train_acc = train_acc
                self.best_train_loss = train_loss
                early_stop_flag = 0

                print(f"Saved new best model at epoch {epoch}")

                state = {
                    **self.configs,
                    "net": self.model.state_dict(),
                    "best_val_loss": self.best_val_loss,
                    "best_val_acc": self.best_val_acc,
                    "best_train_loss": self.best_train_loss,
                    "best_train_acc": self.best_train_acc,
                    "train_loss_list": self.train_loss_list,
                    "val_loss_list": self.val_loss_list,
                    "train_acc_list": self.train_acc_list,
                    "val_acc_list": self.val_acc_list,
                    "optimizer": self.optimizer.state_dict(),
                }

                torch.save(state, self.configs['rs_dir'])
            
            # Create one sample row
            new_row = {"epoch": epoch, "train_acc": round(train_acc, 4), "train_loss": round(train_loss, 4), "val_acc": round(val_acc, 4), "val_loss": round(val_loss, 4), 
                       "topk1train_acc": round(train_acc_dict["1"], 4), "topk3train_acc": round(train_acc_dict["3"], 4), "topk5train_acc": round(train_acc_dict["5"], 4), "topk10train_acc": round(train_acc_dict["10"], 4), "topk20train_acc": round(train_acc_dict["20"], 4),
                       "topk1val_acc": round(val_acc_dict["1"], 4), "topk3val_acc": round(val_acc_dict["3"], 4), "topk5val_acc": round(val_acc_dict["5"], 4), "topk10val_acc": round(val_acc_dict["10"], 4), "topk20val_acc": round(val_acc_dict["20"], 4),
                       "learning_rate": round(current_lr, 7)}
            
            with open(self.configs["tracking_csv"], mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=new_row.keys())
                writer.writerow(new_row)
                
            if log_step != 0 and epoch % log_step == 0:
                line_chart_k_acc(self.configs["tracking_csv"], self.configs["full_rs_dir"], type="train")
                line_chart_k_acc(self.configs["tracking_csv"], self.configs["full_rs_dir"], type="val")
                line_chart(self.configs["tracking_csv"], self.configs["full_rs_dir"], type="train_loss")
                line_chart(self.configs["tracking_csv"], self.configs["full_rs_dir"], type="val_loss")
                line_chart(self.configs["tracking_csv"], self.configs["full_rs_dir"], type="learning_rate")
                
            early_stop_flag += 1
        
            if early_stop_flag > self.configs['earlyStopping']:
                print(f"Early stopping activated at epoch {epoch}")
                break
                
                
                
                
                
                

# if configs['optimizer_chose'] == "RAdam":
        #     print("The selected optimizer is RAdam")
        #     self.optimizer = torch.optim.RAdam(
        #         params = model.parameters(),
        #         lr=configs['lr'])
            
        # if configs['optimizer_chose'] == "WAdam":
        #     print("The selected optimizer is WAdam")
        #     self.optimizer = torch.optim.WAdam(
        #         params = model.parameters(),
        #         weight_decay = configs['weight_decay'],
        #         lr=configs['lr'])
        # Learning Rate Scheduler
        #self.lr_scheduler = None
        # if configs["lr_scheduler"] == "CosineAnnealingLR":
        #     print("The selected Lr Scheduler is CosineAnnealingLR")
        #     self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer = self.optimizer,
        #         T_max=10,     
        #         eta_min=1e-6)