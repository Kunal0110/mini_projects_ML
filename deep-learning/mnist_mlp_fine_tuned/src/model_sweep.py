'''Goal:
Try 3 architectures, train them and record:
number of parameters, test accuracy,
scatter plot of parameters vs accuracy
'''

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  #Flatten 1x28x28 = 784
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#p.numel() counts all elements and requires_grad filters out non-trainable ones    
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataloaders(batch_size = 128, val_frac=0.1):
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    val_size = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl

def accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, actual_output in dataloader:
            input, actual_output = input.to(Device), actual_output.to(Device)
            logits = model(input)
            preds = logits.argmax(dim=1)  #predicted class
            correct += (preds == actual_output).sum().item()
            total += actual_output.size(0)
    return correct / total

#Training single model
def train_one_model(hidden_dim, train_dl, val_dl, test_dl, epochs=10, lr=1e-3):
    model = MLP(hidden_dim=hidden_dim).to(Device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
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
        val_acc = accuracy(model, val_dl)
        print(f"[hidden={hidden_dim}] Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    test_acc = accuracy(model, test_dl)
    return model, test_acc, best_val_acc


def main():
    Path("results").mkdir(exist_ok=True)

    train_dl, val_dl, test_dl = get_dataloaders(batch_size=128)

    configs = [
        (64,  "784-64-64-10"),
        (128, "784-128-128-10"),
        (256, "784-256-256-10"),
    ]

    rows = []
    for hidden_dim, name in configs:
        print(f"\n=== Training model: {name} ===")
        model, test_acc, best_val = train_one_model(
            hidden_dim, train_dl, val_dl, test_dl,
            epochs=10,
            lr=1e-3,
        )
        num_params = count_params(model)

        rows.append({
            "model_name": name,
            "hidden_dim": hidden_dim,
            "num_params": num_params,
            "best_val_acc": best_val,
            "test_acc": test_acc,
        })
    
    csv_path = Path("results") / "model_sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {csv_path}")

    # Scatter plot: parameters vs test accuracy
    num_params = [r["num_params"] for r in rows]
    test_accs = [r["test_acc"] for r in rows]
    labels = [r["model_name"] for r in rows]

    plt.figure()
    plt.scatter(num_params, test_accs)
    for x, y, label in zip(num_params, test_accs, labels):
        plt.text(x, y, label, ha="center", va="bottom")

    plt.xlabel("Number of parameters")
    plt.ylabel("Test accuracy")
    plt.title("Model size vs accuracy (MNIST MLP)")
    plt.xscale("log")  # optional: log-scale for params
    plot_path = Path("results") / "model_size_vs_acc.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot to {plot_path}")

    # Optional JSON dump
    json_path = Path("results") / "model_sweep_results.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()