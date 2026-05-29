import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. Dataset & Preprocessing
# ==========================================
class BilateralFilter(object):
    """
    Custom PyTorch transform to apply a bilateral filter.
    The paper specifies bilateral filtering to minimize noise 
    while preserving edge details of facial features.
    """
    def __init__(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def __call__(self, img):
        img_np = np.array(img)
        filtered = cv2.bilateralFilter(img_np, self.d, self.sigmaColor, self.sigmaSpace)
        return Image.fromarray(filtered)

def get_data_loaders(data_dir, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        BilateralFilter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        BilateralFilter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes

# ==========================================
# 2. Model Architecture
# ==========================================
def get_resnet18_model(num_classes=7):
    """
    Initializes the ResNet18 architecture for Facial Emotion Recognition.
    """
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(model, train_loader, val_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with Accuracy: {best_acc:.4f}")

    print("Training complete.")
    return best_model_path

# ==========================================
# 4. Evaluation & Results Generation
# ==========================================
def evaluate_model(model, val_loader, classes, device, args):
    print("Evaluating model and generating plots...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    num_classes = len(classes)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    metrics_text = f"Accuracy:  {acc * 100:.2f}%\nPrecision: {precision * 100:.2f}%\nRecall:    {recall * 100:.2f}%\nF1-Score:  {f1 * 100:.2f}%\n"
    print(f"\n--- Performance Metrics ---\n{metrics_text}")
    
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(metrics_text)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()

    labels_bin = label_binarize(all_labels, classes=range(num_classes))

    # ROC Curve
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'))
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        precision_c, recall_c, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
        ap = average_precision_score(labels_bin[:, i], all_probs[:, i])
        plt.plot(recall_c, precision_c, lw=2, label=f'{classes[i]} (AP = {ap:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.output_dir, 'precision_recall_curve.png'))
    plt.close()

    print(f"All evaluation results saved to: {os.path.abspath(args.output_dir)}")

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='ResNet18 Facial Emotion Recognition')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./results', help='Folder to save all generated results')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode: train or eval')
    parser.add_argument('--weights', type=str, default='', help='Path to weights for eval mode')
    args = parser.parse_args()

    # Create the output directory to store all generated results
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_data_loaders(args.dataset_path, args.batch_size)
    num_classes = len(classes)
    model = get_resnet18_model(num_classes=num_classes).to(device)

    if args.mode == 'train':
        # Train and then evaluate
        best_model_path = train_model(model, train_loader, val_loader, device, args)
        model.load_state_dict(torch.load(best_model_path))
        evaluate_model(model, val_loader, classes, device, args)
    elif args.mode == 'eval':
        if not args.weights:
            raise ValueError("Please provide --weights path for eval mode.")
        model.load_state_dict(torch.load(args.weights, map_location=device))
        evaluate_model(model, val_loader, classes, device, args)

if __name__ == '__main__':
    main()
