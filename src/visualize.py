"""
Visualize CapsNet reconstructions and capsule dimension perturbations.

Produces figures saved to results/:
  - reconstructions.png  : input vs reconstruction grid
  - perturbations.png    : dimension perturbation grid (Fig. 4 in paper)

Usage:
    python src/visualize.py [--model results/best_capsnet.pt]
                            [--routing 3] [--data-dir data/]
                            [--save-dir results/]
"""

import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from capsnet import CapsNet


def get_args():
    p = argparse.ArgumentParser(description="Visualize CapsNet")
    p.add_argument("--model",    type=str, default="results/best_capsnet.pt")
    p.add_argument("--routing",  type=int, default=3)
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--save-dir", type=str, default="results/")
    return p.parse_args()


@torch.no_grad()
def plot_reconstructions(model, dataset, device, save_path, n=10):
    """Plot input vs reconstructed image pairs."""
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    indices = list(range(n))
    for col, idx in enumerate(indices):
        img, label = dataset[idx]
        img_t = img.unsqueeze(0).to(device)
        _, recon, _ = model(img_t, torch.tensor([label], device=device))
        axes[0, col].imshow(img.squeeze().numpy(), cmap="gray")
        axes[0, col].set_title(str(label), fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(recon.cpu().squeeze().view(28, 28).numpy(), cmap="gray")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=9)
    axes[1, 0].set_ylabel("Recon", fontsize=9)
    plt.suptitle("CapsNet Reconstructions (MNIST Test Set)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved {save_path}")


@torch.no_grad()
def plot_perturbations(model, dataset, device, save_path, digit_class=3):
    """
    Vary each of the 16 DigitCaps dimensions independently over [-0.25, 0.25]
    in 11 steps (Fig. 4 in paper).
    """
    model.eval()
    # Find one image of the target class
    for img, label in dataset:
        if label == digit_class:
            break
    img_t = img.unsqueeze(0).to(device)
    _, _, v = model(img_t, torch.tensor([digit_class], device=device))
    base_vec = v[0, digit_class].clone()   # [16]

    caps_dim = base_vec.size(0)
    perturbations = torch.linspace(-0.25, 0.25, 11)
    fig, axes = plt.subplots(caps_dim, len(perturbations),
                             figsize=(len(perturbations) * 0.9, caps_dim * 0.9))

    for dim in range(caps_dim):
        for col, delta in enumerate(perturbations):
            perturbed = base_vec.clone()
            perturbed[dim] = perturbed[dim] + delta
            # Reconstruct using the decoder directly
            perturbed_v = v.clone()
            perturbed_v[0, digit_class] = perturbed
            recon = model.decoder(perturbed_v, torch.tensor([digit_class], device=device))
            img_np = recon.cpu().view(28, 28).numpy()
            axes[dim, col].imshow(img_np, cmap="gray")
            axes[dim, col].axis("off")

    plt.suptitle(f"Dimension Perturbations — digit '{digit_class}'", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved {save_path}")


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )

    test_ds = datasets.MNIST(
        args.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )

    model = CapsNet(num_routing=args.routing).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model}")

    plot_reconstructions(
        model, test_ds, device,
        os.path.join(args.save_dir, "reconstructions.png")
    )
    for digit in [3, 5, 6]:
        plot_perturbations(
            model, test_ds, device,
            os.path.join(args.save_dir, f"perturbations_digit{digit}.png"),
            digit_class=digit,
        )


if __name__ == "__main__":
    main()
