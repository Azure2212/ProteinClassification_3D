import sys
import os
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

root_project_dir = os.getcwd().split("/")[:4]
root_project_dir = "/".join(root_project_dir)

sys.path.append(os.path.join(root_project_dir, "utils/datasets"))
from pdb_ds import test_tf


def line_chart_k_acc(tracking_csv, path2save, type="train"):
    df = pd.read_csv(tracking_csv)
    plt.figure(figsize=(20, 10)) 
    x = (df['epoch']).tolist()

    # Create the line chart
    plt.plot(x, df[f"topk1{type}_acc"] * 100, marker='o', label='top1')
    plt.plot(x, df[f"topk3{type}_acc"] * 100, marker='o', label='top3')
    plt.plot(x, df[f"topk5{type}_acc"] * 100, marker='o', label='top5')
    plt.plot(x, df[f"topk10{type}_acc"] * 100, marker='o', label='top10')
    plt.plot(x, df[f"topk20{type}_acc"] * 100, marker='o', label='top20')

    # Add labels and title
    plt.xlabel("epoch")
    plt.ylabel("Value(%)")
    plt.title(f"{type.capitalize()} Accuracy with topk")
    plt.legend()
    ticks = list(x[::10])
    if x[-1] not in ticks:
        ticks.append(x[-1])
    plt.grid(True)

    # Save figure instead of showing
    save_path = os.path.join(path2save, f"{type}_topk_accuracy_line_chart.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def line_chart(tracking_csv, path2save, type="train"):
    df = pd.read_csv(tracking_csv)
    plt.figure(figsize=(20, 10)) 
    x = (df['epoch']).tolist()

    # Create the line chart
    y = df[type]
    plt.plot(x, y, marker='o', label=type)

    # Add labels and title
    plt.xlabel("epoch")
    plt.ylabel("Value")
    plt.title(f"{type.capitalize()}")
    plt.legend()
    ticks = list(x[::10])
    if x[-1] not in ticks:
        ticks.append(x[-1])
    
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.7f'))
    plt.grid(True)
    plt.tight_layout()

    # Save figure instead of showing
    save_path = os.path.join(path2save, f"{type}_line_chart.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def realTest_cm(image_size, class_names, checkpoint_path, device, model, path2save, images_per_class, labels_per_class, top_k = 20, saveStatisticsReport=True):

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["net"])
    model.to(device)
    model.eval()
    test_transform = test_tf(image_size)
    # Prepare results
    y_true = []
    y_pred = []

    for i in range(len(images_per_class)):
        img_path = images_per_class[i]
        true_label = labels_per_class[i]

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        img_tensor = test_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # shape: [num_classes]

            # Get top-k indices
            topk_indices = torch.topk(probs, k=top_k).indices.tolist()

            # Accept prediction if true label is in top-k
            if true_label in topk_indices:
                pred_class = true_label
            else:
                pred_class = probs.argmax().item()

        y_true.append(true_label)
        y_pred.append(pred_class)

    # Compute overall accuracy (top-k acceptance)
    total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    topk_acc = total_correct / len(y_true)
    
    print(f"Top-{top_k} acceptance accuracy on real dataset: {topk_acc:.2%}")

    # Optional: Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(list(set(y_true + y_pred)))
    display_labels = [class_names[i] for i in unique_labels]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix (Top-{top_k} acceptance)")
    
    cm_save_path = os.path.join(path2save, f"top{top_k}_confusion_matrix.png")

    # Save figure
    plt.savefig(cm_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_save_path}")
    
    if saveStatisticsReport:

        report_dict = classification_report( y_true, y_pred, target_names=display_labels,
                                            zero_division=0, output_dict=True)

        # Convert to DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        # Save CSV
        csv_path = os.path.join(path2save, "classification_report.csv")
        report_df.to_csv(csv_path, index=True)

        print(f"Classification report saved to: {csv_path}")
    