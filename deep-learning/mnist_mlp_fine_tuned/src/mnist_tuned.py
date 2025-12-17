import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TunedMLP(nn.Module):
    def __init__(self, input_dim =784, hidden_dim = 512, hidden_dim1=256, hidden_dim2 = 128, num_classes=10, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn3 = nn.BatchNorm1d(hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)

        return x
    
    # Weight Initialization (Kaiming)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def get_dataloaders(batch_size=128, val_frac=0.1):
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_full = datasets.MNIST(root='./data', download=True, train=True, transform=transform_train)
    test_ds = datasets.MNIST(root='./data', download=True, train=False, transform=transform_test)

    val_size = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for input, actual_output in dataloader:
            input, actual_output = input.to(Device), actual_output.to(Device)
            logits = model(input)
            loss = criterion(logits, actual_output)
            preds = logits.argmax(dim=1)  #predicted class
            correct += (preds == actual_output).sum().item()
            total += actual_output.size(0)
            loss_sum += loss.item() * actual_output.size(0)

    return {
        "loss" : loss_sum / total,
        "accuracy": correct / total
    }

def train(
    hidden_dim=256, 
    lr= 1e-3,
    max_epochs=50,
    batch_size=128,
    patience=5,
    dropout_p=0.5,
) -> Dict[str, Any]:
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=batch_size)

    model = TunedMLP(hidden_dim=hidden_dim, dropout_p=dropout_p).to(Device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_with_no_improvement = 0

    history = []

    for epoch in range(1, max_epochs+1):
        model.train()
        running_loss = 0.0
        for input, actual_output in train_dl:
            input, actual_output = input.to(Device), actual_output.to(Device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input)
            loss = criterion(logits, actual_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * actual_output.size(0)

        train_loss = running_loss / len(train_dl.dataset)
        val_metrics = evaluate(model, val_dl)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.5f}"
        )

        # Early stopping on validation loss with tiny margin og 1e-4 so microscopic fluctuations don't count
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            epochs_with_no_improvement = 0
        else:
            epochs_with_no_improvement += 1
            if epochs_with_no_improvement >= patience:
                print("Early stopping triggered at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, val_dl)
    final_test = evaluate(model, test_dl)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_metrics_best_epoch": test_metrics,
        "test_loss": final_test["loss"],
        "test_acc": final_test["accuracy"],
    }

def main():
    Path("results").mkdir(exist_ok=True)

    result = train(hidden_dim=256, lr=1e-3, max_epochs=30, patience=5, dropout_p=0.5)

    model = result["model"]
    metrics = {
        "best_epoch": result["best_epoch"],
        "val_metrics_best_epoch": result["val_metrics_best_epoch"],
        "test_loss": result["test_loss"],
        "test_acc": result["test_acc"],
        "history": result["history"],
    }

    # Save the model
    model_path = Path("results") / "mnist_tuned.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved tuned model to {model_path}")

    # Save the metrics
    metrics_path = Path("results") / "mnist_tuned_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved tuned model metrics to {metrics_path}")

    print(f"Final test accuracy: {metrics['test_acc']: .4f}")

if __name__ == "__main__":
    main()