"""
Efficiency / timing test for CapsNet.

Reports:
  - Training time per epoch on small subset (1,000 samples)
  - Inference time for full test set
  - Throughput (samples/sec)

Usage:
    python src/timing_test.py [--routing 3] [--data-dir data/] [--save-dir results/]
"""

import argparse
import os
import time
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from capsnet import CapsNet, total_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--routing",    type=int, default=3)
    p.add_argument("--data-dir",   type=str, default="data/")
    p.add_argument("--save-dir",   type=str, default="results/")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--small-n",    type=int, default=1000, help="Small subset size")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(args.data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)

    # Small subset for training timing
    small_train = Subset(train_ds, list(range(args.small_n)))
    small_loader = DataLoader(small_train, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,    batch_size=args.batch_size, shuffle=False,
                              num_workers=2)

    model = CapsNet(num_routing=args.routing).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Training timing (1 epoch on small dataset) ---
    model.train()
    t0 = time.time()
    for images, labels in small_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        v_lengths, reconstruction, _ = model(images, labels)
        loss, _, _ = total_loss(v_lengths, labels, reconstruction, images)
        loss.backward()
        optimizer.step()
    train_time = time.time() - t0
    train_throughput = args.small_n / train_time

    print(f"\n--- Training Timing (n={args.small_n}) ---")
    print(f"  Epoch time    : {train_time:.2f}s")
    print(f"  Throughput    : {train_throughput:.1f} samples/sec")

    # --- Inference timing (full test set) ---
    model.eval()
    n_test = len(test_ds)
    t0 = time.time()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            model(images)
    infer_time = time.time() - t0
    infer_throughput = n_test / infer_time

    print(f"\n--- Inference Timing (n={n_test}) ---")
    print(f"  Total time    : {infer_time:.2f}s")
    print(f"  Throughput    : {infer_throughput:.1f} samples/sec")
    print(f"  Latency       : {infer_time/n_test*1000:.3f} ms/sample")

    results = {
        "device": str(device),
        "routing_iterations": args.routing,
        "small_train_n": args.small_n,
        "train_epoch_time_s": train_time,
        "train_throughput_samples_per_sec": train_throughput,
        "test_n": n_test,
        "inference_time_s": infer_time,
        "inference_throughput_samples_per_sec": infer_throughput,
        "inference_latency_ms_per_sample": infer_time / n_test * 1000,
    }
    out_path = os.path.join(args.save_dir, "timing_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTiming results saved to {out_path}")


if __name__ == "__main__":
    main()
