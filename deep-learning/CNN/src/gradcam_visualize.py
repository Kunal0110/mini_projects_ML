import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Config

CHECKPOINT_PATH = "best_transfer_resnet18.pth"   # from your fine-tuning step
NUM_IMAGES = 8                                   # how many misclassified examples to visualize
BATCH_SIZE = 64
RANDOM_SEED = 42
OUTPUT_FILE = "gradcam_examples.png"

# 2. Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

# 3. Data: CIFAR-10 test set with SAME transforms as training

def get_test_loader(batch_size: int) -> Tuple[DataLoader, List[str]]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=test_transform,
        download=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    class_names = test_dataset.classes  # ['airplane', 'automobile', ...]
    return test_loader, class_names


# 4. Model: same architecture, load fine-tuned weights

def build_model(num_classes: int = 10) -> nn.Module:
    # Architecture must match the one used for training.
    # We don’t freeze anything here; we want gradients for Grad-CAM.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_finetuned_model(device: torch.device) -> nn.Module:
    model = build_model(num_classes=10)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# 5. Utility: unnormalize image for visualization

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    """
    tensor: shape [3, H, W], normalized with ImageNet mean/std.
    Returns: numpy float32 image in [0,1], shape [H, W, 3].
    """
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)


# 6. Collect misclassified images

def collect_misclassified_examples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_examples: int
):
    """
    Returns lists of (image_tensor, true_label, pred_label) for misclassified samples.
    image_tensor is the normalized tensor as seen by the model.
    """
    misclassified = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            mismatches = preds != labels
            if mismatches.any():
                indices = torch.where(mismatches)[0]
                for idx in indices:
                    img = images[idx].cpu()     # keep tensor on CPU for later
                    true_label = labels[idx].item()
                    pred_label = preds[idx].item()
                    misclassified.append((img, true_label, pred_label))

                    if len(misclassified) >= max_examples:
                        return misclassified

    return misclassified


# 7. Grad-CAM visualization

def generate_gradcam_visualizations(
    model: nn.Module,
    device: torch.device,
    misclassified_examples,
    class_names: List[str]
):
    """
    For each misclassified example:
      - Run Grad-CAM on the predicted (wrong) class.
      - Overlay heatmap on the RGB image.
    Returns a list of (rgb_image, heatmap_overlay, true_label, pred_label).
    """
    # For ResNet18 the recommended target layer is model.layer4[-1] (last conv block)
    target_layers = [model.layer4[-1]]

    visualizations = []

    use_cuda = device.type == "cuda"
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for img_tensor, true_label, pred_label in misclassified_examples:
            # Model input: add batch dimension and send to device
            input_tensor = img_tensor.unsqueeze(0).to(device)

            # We explain the predicted (wrong) class:
            targets = [ClassifierOutputTarget(pred_label)]

            # grayscale_cam: [batch, H, W] -> here [1, H, W]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # Prepare RGB image in [0,1] for overlay
            rgb_img = tensor_to_rgb_image(img_tensor)

            # Overlay heatmap on original RGB image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            visualizations.append(
                (rgb_img, cam_image, true_label, pred_label)
            )

    return visualizations


# 8. Make a grid and save as gradcam_examples.png

def save_grid(visualizations, class_names: List[str], output_file: str):
    """
    Creates a matplotlib grid: for each misclassified sample, show
    original image (left) + Grad-CAM overlay (right) with labels.
    """
    num_examples = len(visualizations)
    cols = 2      # original + heatmap
    rows = num_examples

    fig, axes = plt.subplots(rows, cols, figsize=(6, 3 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # make it 2D for uniform indexing

    for i, (rgb_img, cam_img, true_label, pred_label) in enumerate(visualizations):
        # Original
        ax_orig = axes[i, 0]
        ax_orig.imshow(rgb_img)
        ax_orig.axis("off")
        ax_orig.set_title(f"Original\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}")

        # Grad-CAM overlay
        ax_cam = axes[i, 1]
        ax_cam.imshow(cam_img)
        ax_cam.axis("off")
        ax_cam.set_title("Grad-CAM")

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Saved Grad-CAM grid to {output_file}")


# 9. Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader, class_names = get_test_loader(BATCH_SIZE)
    model = load_finetuned_model(device)

    print("Collecting misclassified examples...")
    misclassified_examples = collect_misclassified_examples(
        model, test_loader, device, max_examples=NUM_IMAGES
    )

    if len(misclassified_examples) == 0:
        print("No misclassified examples found. (Model might be too good on this subset!)")
        return

    print(f"Found {len(misclassified_examples)} misclassified examples. Generating Grad-CAMs...")
    visualizations = generate_gradcam_visualizations(
        model, device, misclassified_examples, class_names
    )

    save_grid(visualizations, class_names, OUTPUT_FILE)


if __name__ == "__main__":
    main()
