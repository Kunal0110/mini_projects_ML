import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 1. Configuration / Hyperparameters

BATCH_SIZE = 128           # How many images per mini-batch
NUM_EPOCHS = 20           
LEARNING_RATE = 1e-4       # Small LR for fine-tuning
WEIGHT_DECAY = 1e-4        # L2 regularization strength
MOMENTUM = 0.9             # Momentum for SGD
VAL_SPLIT = 0.1            # 10% of training data used as validation
RANDOM_SEED = 42
NUM_WORKERS = 4            # DataLoader workers (tune based on CPU)
PIN_MEMORY = True          # Speed-up when using GPU
LOG_FILE = "transfer_resnet18_log.csv"

# 2. Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# 3. Data transforms & loaders
# Note: ResNet18 expects 224x224 and ImageNet normalization.

def get_data_loaders(batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # ImageNet normalization values (used during original ResNet training)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),                # Slightly larger resize
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),           # Random crop to 224x224
        transforms.RandomHorizontalFlip(),    # Simple augmentation
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.25)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),           # Deterministic crop for eval
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Download CIFAR-10
    train_full = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=train_transform,
        download=True
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=test_transform,
        download=True
    )

    # Split training set into train + val
    num_train = len(train_full)
    num_val = int(val_split * num_train)
    num_train = num_train - num_val
    train_dataset, val_dataset = random_split(
        train_full,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, test_loader

# 4. Build & configure ResNet18

def build_model(num_classes: int = 10) -> nn.Module:
    """
    Load a pretrained ResNet18, replace the final FC layer, and freeze backbone.
    """

    # For newer torchvision: use weights API
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # For compatibility with older versions:
    # model = models.resnet18(pretrained=True)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    for param in model.layer3.parameters():
        param.requires_grad = True

    return model

# 5. Training & evaluation loops

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)              # Shape: [batch_size, 10]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Compute accuracy
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# 6. Main training routine

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT
    )

    model = build_model(num_classes=10)
    model = model.to(device)

    # Only train parameters that require grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in params_to_update)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        params_to_update,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Simple LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min= 1e-6
    )

    # Prepare log file
    with open(LOG_FILE, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}, LR={current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_model(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Log to CSV
        with open(LOG_FILE, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_acc:.4f},"
                f"{val_loss:.4f},{val_acc:.4f},{current_lr:.6f}\n"
            )

        # Keep track of best val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_transfer_resnet18.pth")

        scheduler.step()

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load("best_transfer_resnet18.pth", map_location=device))
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # You should see a big jump vs training from scratch
    # 80–85%+ is very reasonable with just a few epochs of transfer learning.


if __name__ == "__main__":
    main()
