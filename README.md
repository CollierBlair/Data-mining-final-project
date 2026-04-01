# Data Mining Final Project — CapsNet: Dynamic Routing Between Capsules

Implementation of **"Dynamic Routing Between Capsules"** (Sabour, Frosst, Hinton — NIPS 2017) for a data mining course project.

> Paper: https://arxiv.org/abs/1710.09829

---

## Problem Definition

**Task:** Multi-class image classification on handwritten digits (MNIST).

Standard convolutional networks discard spatial relationships between features via max-pooling. Capsule Networks replace scalar neuron outputs with **vector-valued capsules** that encode both the presence *and* pose (position, orientation, scale) of an entity. A dynamic **routing-by-agreement** mechanism replaces max-pooling: lower-level capsules route their output to the higher-level capsule whose activity vector best agrees with their prediction.

**Dataset:** MNIST — 60,000 training images and 10,000 test images of 28×28 grayscale handwritten digits (10 classes, 0–9).

---

## Algorithm

### CapsNet Architecture (3 layers)

```
Input [B, 1, 28, 28]
  ↓
Conv1          256 filters, 9×9, stride 1, ReLU  → [B, 256, 20, 20]
  ↓
PrimaryCaps    32 types × 8D capsules, 9×9 conv stride 2 → [B, 1152, 8]
               squash nonlinearity applied
  ↓
DigitCaps      10 capsules (one per class) × 16D
               Dynamic routing-by-agreement (r=3 iterations) → [B, 10, 16]
  ↓
Decoder        FC 512 → 1024 → 784 (sigmoid) — reconstruction regularizer
```

### Key Equations

**Squashing (Eq. 1):**
$$v_j = \frac{||s_j||^2}{1+||s_j||^2} \cdot \frac{s_j}{||s_j||}$$

**Prediction vectors (Eq. 2):**
$$\hat{u}_{j|i} = W_{ij} u_i, \quad s_j = \sum_i c_{ij} \hat{u}_{j|i}$$

**Routing softmax (Eq. 3):**
$$c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}$$

**Margin loss (Eq. 4):**
$$L_k = T_k \max(0, m^+ - ||v_k||)^2 + \lambda(1-T_k)\max(0, ||v_k|| - m^-)^2$$

Parameters: $m^+=0.9$, $m^-=0.1$, $\lambda=0.5$, reconstruction scale $=0.0005$.

---

## Requirements

```
Python 3.8+
torch >= 2.0.0
torchvision >= 0.15.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
numpy >= 1.24.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Train

```bash
python src/train.py \
    --epochs 50 \
    --batch-size 128 \
    --routing 3 \
    --lr 1e-3 \
    --data-dir data/ \
    --save-dir results/
```

The best model checkpoint is saved to `results/best_capsnet.pt`. Training history is written to `results/train_history.json`.

Quick smoke-test (5 epochs):

```bash
python src/train.py --epochs 5
```

### 2. Evaluate (Effectiveness Test)

```bash
python src/evaluate.py \
    --model results/best_capsnet.pt \
    --routing 3 \
    --data-dir data/ \
    --save-dir results/
```

Outputs **Accuracy**, **Macro-F1**, and **Micro-F1** on the 10,000-sample MNIST test set, plus a per-class classification report. Results written to `results/eval_results.json`.

### 3. Efficiency / Timing Test

```bash
python src/timing_test.py \
    --routing 3 \
    --data-dir data/ \
    --save-dir results/ \
    --small-n 1000
```

Reports training throughput (samples/sec) on a 1,000-sample subset and full-test-set inference latency (ms/sample). Written to `results/timing_results.json`.

### 4. Visualize Reconstructions & Dimension Perturbations

```bash
python src/visualize.py \
    --model results/best_capsnet.pt \
    --routing 3 \
    --data-dir data/ \
    --save-dir results/
```

Saves:
- `results/reconstructions.png` — input vs. reconstruction pairs
- `results/perturbations_digit3.png`, `_digit5.png`, `_digit6.png` — capsule dimension perturbation grids (replicating Fig. 4 of the paper)

---

## Evaluation Measures

| Measure | Description |
|---|---|
| **Accuracy / Error Rate** | Primary metric (matches the paper's Table 1) |
| **Macro-F1** | Unweighted mean F1 across all 10 digit classes |
| **Micro-F1** | Global F1 aggregated over all instances (= Accuracy for single-label) |
| **Inference Time** | Total and per-sample latency on the test set |
| **Training Time** | Time per epoch on a 1,000-sample subset |

---

## Expected Results

Paper reports **0.25% test error** (99.75% accuracy) with 3 routing iterations + reconstruction regularizer on a 3-layer CapsNet. This implementation targets the same configuration using PyTorch.

---

## Project Structure

```
Data-mining-final-project/
├── README.md
├── requirements.txt
├── data/               ← MNIST downloaded here automatically
├── results/            ← checkpoints, JSON metrics, figures
└── src/
    ├── capsnet.py      ← model definition (CapsNet, routing, losses)
    ├── train.py        ← training loop
    ├── evaluate.py     ← effectiveness evaluation (Accuracy, Macro-F1, Micro-F1)
    ├── timing_test.py  ← efficiency / timing benchmarks
    └── visualize.py    ← reconstruction & dimension perturbation figures
```

---

## Extensions and Improvements

1. **MultiMNIST** — overlay two digits (80% bounding-box overlap) to test segmentation capability; the paper achieves 5.2% error vs. 8.1% for a CNN baseline.
2. **affNIST** — test robustness to affine transformations without retraining.
3. **Deeper capsule hierarchy** — add intermediate capsule layers for richer part-whole relationships.
4. **EM Routing** — replace dynamic routing with the expectation-maximization variant (Matrix Capsules, 2018).
5. **Graph datasets** — adapt capsule routing to graph-structured data using the provided CA-GrQc or DBLP co-authorship graphs.

---

## References

Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules. *Advances in Neural Information Processing Systems (NIPS 2017)*. https://arxiv.org/abs/1710.09829
