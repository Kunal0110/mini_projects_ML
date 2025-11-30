import os
import json
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Utility functions
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
    return 100.0 * correct / total

# Improved MLP with regularization and better architecture
class MNISTMLP(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)
    
#Early stopping helper
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
# Train and Validation loops

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Main training logic

def main():
    device = get_device()
    print("Using device:", device)

    #Tensorboard writer
    writer = SummaryWriter(log_dir="runs/mnist_mlp")

    # Enhanced data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets with augmentation for training
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Apply test transform to validation set
    val_dataset.dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=test_transform)

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # Improved model and training setup
    model = MNISTMLP(dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    
    early_stopper = EarlyStopping(patience=7, min_delta=0.001)
    best_state_dict = None
    best_val_acc = 0.0
    best_epoch = 0

    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step LR scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Tensorboard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Track best model
        if val_acc > best_val_acc:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
            best_epoch = epoch

        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Best model from epoch {best_epoch} with validation accuracy {best_val_acc:.2f}%")

    # Final metrics on train/val/test with best model
    train_loss_final, train_acc_final = evaluate(model, train_loader, criterion, device)
    val_loss_final, val_acc_final = evaluate(model, val_loader, criterion, device)
    test_loss_final, test_acc_final = evaluate(model, test_loader, criterion, device)

    print(f"Final Metrics:")
    print(f"Train Loss: {train_loss_final:.4f} | Train Acc: {train_acc_final:.2f}%")
    print(f"Val Loss: {val_loss_final:.4f} | Val Acc: {val_acc_final:.2f}%")
    print(f"Test Loss: {test_loss_final:.4f} | Test Acc: {test_acc_final:.2f}%")

    # Save model

    model_path = "mnist_mlp.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save metrics to JSON
    metrics = {
        "train_loss": train_loss_final,
        "train_acc": train_acc_final,
        "val_loss": val_loss_final,
        "val_acc": val_acc_final,
        "test_loss": test_loss_final,
        "test_acc": test_acc_final,
        "best_epoch": best_epoch
    }

    metrics_path = "mnist_mlp_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    writer.close()

if __name__ == "__main__":
    main()