"""
Evaluation script for CapsNet on MNIST.

Computes:
  - Test Accuracy  (paper metric)
  - Macro-F1       (alternative measure per project requirements)
  - Micro-F1       (alternative measure per project requirements)
  - Inference time (efficiency test)

Usage:
    python src/evaluate.py [--model results/best_capsnet.pt]
                           [--routing 3] [--data-dir data/]
                           [--save-dir results/]
"""

import argparse
import os
import time
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from capsnet import CapsNet


def get_args():
    p = argparse.ArgumentParser(description="Evaluate CapsNet on MNIST")
    p.add_argument("--model",    type=str, default="results/best_capsnet.pt")
    p.add_argument("--routing",  type=int, default=3)
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--save-dir", type=str, default="results/")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    start = time.time()
    for images, labels in loader:
        images = images.to(device)
        v_lengths, _, _ = model(images)
        preds = v_lengths.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    elapsed = time.time() - start
    return all_preds, all_labels, elapsed


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    test_ds = datasets.MNIST(
        args.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = CapsNet(num_routing=args.routing).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model}")

    preds, labels, inference_time = run_inference(model, test_loader, device)

    accuracy  = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro")
    micro_f1  = f1_score(labels, preds, average="micro")

    print("\n" + "="*55)
    print("  EVALUATION RESULTS ON MNIST TEST SET (10,000 samples)")
    print("="*55)
    print(f"  Accuracy       : {accuracy*100:.2f}%  (error: {(1-accuracy)*100:.2f}%)")
    print(f"  Macro-F1       : {macro_f1:.4f}")
    print(f"  Micro-F1       : {micro_f1:.4f}")
    print(f"  Inference time : {inference_time:.2f}s  "
          f"({inference_time/len(labels)*1000:.2f} ms/sample)")
    print("="*55)

    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=[str(i) for i in range(10)]))

    results = {
        "accuracy":        accuracy,
        "test_error_pct":  (1 - accuracy) * 100,
        "macro_f1":        macro_f1,
        "micro_f1":        micro_f1,
        "inference_time_s": inference_time,
        "ms_per_sample":   inference_time / len(labels) * 1000,
        "num_test_samples": len(labels),
    }
    out_path = os.path.join(args.save_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
