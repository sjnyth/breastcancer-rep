from __future__ import annotations

from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns
import matplotlib.pyplot as plt

DATA_ROOT = Path("dataset_splits")  # update to GPU workspace path
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"
CHECKPOINT_DIR = Path("checkpoints_convnext_stage")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = CHECKPOINT_DIR / "training_metrics_stage.csv"
if METRICS_PATH.exists():
    METRICS_PATH.unlink()

IMAGE_SIZE = 384
BATCH_SIZE = 64  # adjust for higher resolution
HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 40
LR_HEAD = 1e-3
LR_FINE = 1e-4
STEP_SIZE = 7
GAMMA = 0.1
SUBSET_SIZE = None  # set to e.g. 200 for faster experiments

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=transform_train)
val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=transform_eval)
test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=transform_eval)

if SUBSET_SIZE is not None:
    train_dataset = Subset(train_dataset, range(min(SUBSET_SIZE, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(SUBSET_SIZE, len(val_dataset))))
    test_dataset = Subset(test_dataset, range(min(SUBSET_SIZE, len(test_dataset))))

train_targets = []
for _, labels in DataLoader(train_dataset, batch_size=1, shuffle=False):
    train_targets.extend(labels.tolist())
class_counts = Counter(train_targets)
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}
sample_weights = torch.tensor([class_weights[label] for label in train_targets], dtype=torch.double)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.feature_extractor = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=[2, 3])
        return self.head(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets):
        base_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        focal_factor = torch.where(targets == 1, (1 - probs) ** self.gamma, probs ** self.gamma)
        loss = self.alpha * focal_factor * base_loss
        return loss.mean()


def compute_pos_weight(counts: Counter):
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(train_mode):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze()
            loss = criterion(logits, labels.float())

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples, total_loss / len(loader)


def evaluate(model, loader, threshold=0.5):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze()
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(lbls.numpy())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    return dict(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds),
        recall=recall_score(labels, preds),
        f1=f1_score(labels, preds),
        roc_auc=roc_auc_score(labels, probs),
        specificity=specificity,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtTiny().to(device)
pos_weight = compute_pos_weight(class_counts).to(device)
criterion = FocalLoss(gamma=2.0, alpha=1.0, pos_weight=pos_weight)

# Stage 1: freeze backbone, train head
train_losses_stage1, val_losses_stage1 = [], []
train_losses_stage2, val_losses_stage2 = [], []

for param in model.feature_extractor.parameters():
    param.requires_grad = False
optimizer = optim.Adam(model.head.parameters(), lr=LR_HEAD)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

for epoch in range(HEAD_EPOCHS):
    train_acc, train_loss = run_epoch(model, train_loader, criterion, optimizer)
    val_acc, val_loss = run_epoch(model, val_loader, criterion)
    scheduler.step()
    train_losses_stage1.append(train_loss)
    val_losses_stage1.append(val_loss)
    print(f"Stage1 Epoch {epoch+1}/{HEAD_EPOCHS} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Stage 2: unfreeze all layers, lower LR
for param in model.feature_extractor.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=LR_FINE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

best_val = 0.0
best_ckpt = CHECKPOINT_DIR / "model_best_stage.pth"
early_stop_patience = 7
epochs_no_improve = 0

for epoch in range(FINETUNE_EPOCHS):
    train_acc, train_loss = run_epoch(model, train_loader, criterion, optimizer)
    val_acc, val_loss = run_epoch(model, val_loader, criterion)
    scheduler.step()
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), best_ckpt)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    train_losses_stage2.append(train_loss)
    val_losses_stage2.append(val_loss)
    print(f"Stage2 Epoch {epoch+1}/{FINETUNE_EPOCHS} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if epochs_no_improve >= early_stop_patience:
        print("Early stopping triggered.")
        break

if not best_ckpt.exists():
    raise FileNotFoundError("No checkpoint saved")
model.load_state_dict(torch.load(best_ckpt, map_location=device))
metrics = evaluate(model, test_loader)
print("Test metrics:", metrics)

# Confusion matrix and ROC curve
labels, probs = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze()
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(lbls.numpy())
labels = np.array(labels)
probs = np.array(probs)
preds = (probs > 0.5).astype(int)
conf_matrix = confusion_matrix(labels, preds)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("ConvNeXt Staged Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0.5, 1.5], ["Benign", "Malignant"])
plt.yticks([0.5, 1.5], ["Benign", "Malignant"])
plt.savefig(CHECKPOINT_DIR / "focal_loss_confusion_matrix_stage64.pdf", dpi=600)
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc_val = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_val:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ConvNeXt Staged ROC Curve")
plt.legend(loc="lower right")
plt.savefig(CHECKPOINT_DIR / "focal_loss_roc_curve_stage64.pdf", dpi=600)
plt.show()

# Loss curves
plt.figure()
plt.plot(range(1, len(train_losses_stage1) + 1), train_losses_stage1, label="Stage1 Train Loss")
plt.plot(range(1, len(val_losses_stage1) + 1), val_losses_stage1, label="Stage1 Val Loss")
plt.plot(range(1, len(train_losses_stage2) + 1), train_losses_stage2, label="Stage2 Train Loss")
plt.plot(range(1, len(val_losses_stage2) + 1), val_losses_stage2, label="Stage2 Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig(CHECKPOINT_DIR / "loss_curves_absolute.pdf", dpi=600)
plt.show()
