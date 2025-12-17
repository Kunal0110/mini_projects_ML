import os
import csv
import random
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse

import matplotlib.pyplot as plt

# Configuration / Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 4
BASELINE_LOG = "cifar_train_log_baseline.csv"
AUG_LOG = "cifar_train_log_aug.csv"
PLOT_FILE = "cifar_accuracy_compare.png"

# Reproducibility
def set_seed(seed: int =42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #Make cuDNN behave in a deterministic way
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# CNN model definition
# Conv(3 -> 32, 3x3) -> ReLU -> MaxPool
# Conv(32→64, 3x3) → ReLU → MaxPool
#  FC layers → 10 outputs

class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCIFAR10CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
    
# Transformations for CIFAR10
def get_transforms(use_augmentation: bool):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tfms = []

    if use_augmentation:
        train_tfms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
        ])

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_transform = transforms.Compose(train_tfms)

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, eval_transform

# Data loading for CIFAR10
def get_dataloaders(batch_size: int, val_split: float, num_workers: int, seed: int, use_augmentation: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Returns train, val and test dataloaders for CIFAR10 dataset

    train_transform, val_transform = get_transforms(use_augmentation)
    # Download / load CIFAR10
    trainval_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainval_dataset_for_val = datasets.CIFAR10(root="./data",train=True,download=False,transform=val_transform)  # deterministic for val
    test_dataset = datasets.CIFAR10(root='./data', download=True, train=False, transform=val_transform)

    # Split train into train + val
    total_train_samples = len(trainval_dataset)
    val_size = int(total_train_samples * val_split)
    train_size = total_train_samples - val_size
    train_subset, val_subset_indices = random_split(list(range(total_train_samples)), [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_indices = train_subset.indices
    val_indices = val_subset_indices.indices

    from torch.utils.data import Subset
    train_dataset = Subset(trainval_dataset, train_indices)
    val_dataset = Subset(trainval_dataset_for_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# Training and evaluation helpers
def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler=None
) -> Tuple[float, float]:
    # Retrun average loss and accuracy for the epoch
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Accumulate stats
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, dim=1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / total_predictions
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    # Return average loss and accuracy for the epoch
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / total_predictions
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Training loop with CSV logging
def train_model(
      *,
    log_file: str,
    use_augmentation: bool,
    batch_size: int,
    num_epochs: int,
    lr: float,
    val_split: float,
    num_workers: int,
    seed: int  
) -> Dict[str, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data
    train_laoder, val_loader, test_loader = get_dataloaders(batch_size=batch_size, val_split= val_split, num_workers= num_workers, seed=seed, use_augmentation=use_augmentation)

    # Initialize model, loss, and optimizer
    model = SimpleCIFAR10CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_laoder))

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # Prepare CSV logging
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Test Accuracy', 'Test loss'])

        # Training loop
        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = train_one_epoch(model, train_laoder, criterion, optimizer, device, scheduler)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
            print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}"
        )
            # Log to CSV
            writer.writerow([epoch, train_loss, train_accuracy, val_loss, val_accuracy, test_accuracy, test_loss])

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_accuracy)
            history["val_acc"].append(val_accuracy)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_accuracy) 

            tag = "AUG" if use_augmentation else "BASE"

    return history

def plot_accuracy_comparison(
    baseline_hist: Dict[str, List[float]],
    aug_hist: Dict[str, List[float]],
    out_file: str
):
    plt.figure()
    plt.plot(baseline_hist["epoch"], baseline_hist["train_acc"], label="Train (baseline)")
    plt.plot(baseline_hist["epoch"], baseline_hist["val_acc"], label="Val (baseline)")
    plt.plot(aug_hist["epoch"], aug_hist["train_acc"], label="Train (aug)")
    plt.plot(aug_hist["epoch"], aug_hist["val_acc"], label="Val (aug)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Accuracy: Baseline vs Augmented")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# -----------------------------
# 9. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN baseline vs augmentation experiment.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    # Modes
    parser.add_argument("--run-baseline-only", action="store_true")
    parser.add_argument("--run-aug-only", action="store_true")
    parser.add_argument("--run-both", action="store_true")

    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    set_seed(args.seed)

    # Decide mode
    if not (args.run_baseline_only or args.run_aug_only or args.run_both):
        # default behavior: run both
        args.run_both = True

    baseline_hist = None
    aug_hist = None

    if args.run_baseline_only or args.run_both:
        baseline_hist = train_model(
            log_file=BASELINE_LOG,
            use_augmentation=False,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed
        )

    if args.run_aug_only or args.run_both:
        aug_hist = train_model(
            log_file=AUG_LOG,
            use_augmentation=True,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed
        )

    # Plot if both are available
    if baseline_hist is not None and aug_hist is not None:
        plot_accuracy_comparison(baseline_hist, aug_hist, PLOT_FILE)
        print(f"Saved comparison plot to: {PLOT_FILE}")
        print(f"Baseline log: {BASELINE_LOG}")
        print(f"Aug log: {AUG_LOG}")

        # Quick console summary
        print("\n--- Quick Comparison (last epoch) ---")
        print(f"Baseline Train Acc: {baseline_hist['train_acc'][-1]:.4f}")
        print(f"Baseline Val Acc  : {baseline_hist['val_acc'][-1]:.4f}")
        print(f"Aug Train Acc     : {aug_hist['train_acc'][-1]:.4f}")
        print(f"Aug Val Acc       : {aug_hist['val_acc'][-1]:.4f}")

    else:
        # Still print where logs are
        if baseline_hist is not None:
            print(f"Baseline log: {BASELINE_LOG}")
        if aug_hist is not None:
            print(f"Aug log: {AUG_LOG}")


if __name__ == "__main__":
    main()