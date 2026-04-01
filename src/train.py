"""
Training script for CapsNet on MNIST.

Usage:
    python src/train.py [--epochs 50] [--batch-size 128] [--routing 3]
                        [--lr 1e-3] [--save-dir results/]
"""

import argparse
import os
import time
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from capsnet import CapsNet, total_loss


def get_args():
    p = argparse.ArgumentParser(description="Train CapsNet on MNIST")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--routing",    type=int,   default=3,    help="Routing iterations")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--save-dir",   type=str,   default="results/")
    p.add_argument("--data-dir",   type=str,   default="data/")
    return p.parse_args()


def get_loaders(data_dir, batch_size):
    # Paper: 2-pixel shift augmentation only
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()

    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        v_lengths, reconstruction, _ = model(images, labels)
        loss, _, _ = total_loss(v_lengths, labels, reconstruction, images)
        loss.backward()
        optimizer.step()

        preds = v_lengths.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        v_lengths, reconstruction, _ = model(images)
        loss, _, _ = total_loss(v_lengths, labels, reconstruction, images)

        preds = v_lengths.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    model = CapsNet(num_routing=args.routing).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_capsnet.pt"))

    total_train_time = time.time() - train_start
    print(f"\nTotal training time: {total_train_time:.1f}s")
    print(f"Best test accuracy:  {best_acc:.4f}  ({(1-best_acc)*100:.2f}% error)")

    # Save history and timing
    history["total_train_time_s"] = total_train_time
    history["best_test_acc"] = best_acc
    history["num_params"] = num_params
    with open(os.path.join(args.save_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
