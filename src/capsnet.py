"""
CapsNet implementation based on:
  "Dynamic Routing Between Capsules" - Sabour, Frosst, Hinton (NIPS 2017)
  https://arxiv.org/abs/1710.09829

Architecture:
  Conv1 -> PrimaryCaps -> DigitCaps -> Decoder (reconstruction regularizer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s, dim=-1):
    """Non-linear squashing function (Eq. 1 in paper)."""
    norm_sq = (s ** 2).sum(dim=dim, keepdim=True)
    norm = norm_sq.sqrt()
    return (norm_sq / (1.0 + norm_sq)) * (s / (norm + 1e-8))


class PrimaryCaps(nn.Module):
    """
    Primary capsule layer.
    Produces [B, num_caps, caps_dim] capsule output vectors.
    """
    def __init__(self, in_channels=256, num_capsules=32, caps_dim=8, kernel_size=9, stride=2):
        super().__init__()
        # Each of the 32 capsule types is an 8-channel conv
        self.caps_dim = caps_dim
        self.num_capsules = num_capsules
        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * caps_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        # x: [B, 256, 20, 20]  →  out: [B, 32*8, 6, 6]
        out = self.conv(x)                          # [B, 256, 6, 6]
        B, _, H, W = out.shape
        # Reshape to [B, num_capsules * H * W, caps_dim]
        out = out.view(B, self.num_capsules, self.caps_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()   # [B, 32, 6, 6, 8]
        out = out.view(B, -1, self.caps_dim)             # [B, 1152, 8]
        return squash(out)


class DigitCaps(nn.Module):
    """
    Digit capsule layer with dynamic routing-by-agreement (Procedure 1).
    """
    def __init__(self, in_caps=1152, in_dim=8, num_classes=10, caps_dim=16, num_routing=3):
        super().__init__()
        self.in_caps = in_caps
        self.num_classes = num_classes
        self.caps_dim = caps_dim
        self.num_routing = num_routing

        # Weight matrices W_ij  shape: [1, in_caps, num_classes, caps_dim, in_dim]
        self.W = nn.Parameter(
            torch.randn(1, in_caps, num_classes, caps_dim, in_dim) * 0.01
        )

    def forward(self, u):
        """
        u: [B, in_caps, in_dim]
        returns v: [B, num_classes, caps_dim]
        """
        B = u.size(0)

        # Expand u for matmul: [B, in_caps, 1, in_dim, 1]
        u_hat = u.unsqueeze(2).unsqueeze(4)          # [B, 1152, 1, 8, 1]
        W = self.W.expand(B, -1, -1, -1, -1)         # [B, 1152, 10, 16, 8]

        # Prediction vectors u_hat_j|i = W_ij * u_i
        u_hat = torch.matmul(W, u_hat).squeeze(-1)   # [B, 1152, 10, 16]

        # Detach u_hat from graph during routing iterations (no grad through b)
        u_hat_detached = u_hat.detach()

        # Initial logits b_ij = 0
        b = torch.zeros(B, self.in_caps, self.num_classes, 1, device=u.device)

        for iteration in range(self.num_routing):
            c = F.softmax(b, dim=2)                  # [B, 1152, 10, 1]

            # Use detached predictions for all but the last iteration
            if iteration < self.num_routing - 1:
                s = (c * u_hat_detached).sum(dim=1)  # [B, 10, 16]
                v = squash(s, dim=-1)                # [B, 10, 16]
                # Agreement: a_ij = v_j · u_hat_j|i
                v_exp = v.unsqueeze(1)               # [B, 1, 10, 16]
                agreement = (u_hat_detached * v_exp).sum(dim=-1, keepdim=True)  # [B, 1152, 10, 1]
                b = b + agreement
            else:
                # Final iteration: use actual u_hat so gradients flow
                s = (c * u_hat).sum(dim=1)
                v = squash(s, dim=-1)

        return v   # [B, 10, 16]


class Decoder(nn.Module):
    """
    Reconstruction decoder (Fig. 2 in paper).
    3 FC layers: 512 → 1024 → 784 (sigmoid output).
    """
    def __init__(self, caps_dim=16, num_classes=10, img_size=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(caps_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_size),
            nn.Sigmoid(),
        )

    def forward(self, v, labels):
        """
        v:      [B, 10, 16]  digit capsule vectors
        labels: [B]          ground-truth class indices (one-hot mask)
        """
        B = v.size(0)
        # Mask out all but the correct capsule
        mask = F.one_hot(labels, num_classes=v.size(1)).float()  # [B, 10]
        mask = mask.unsqueeze(2)                                   # [B, 10, 1]
        masked = (v * mask).view(B, -1)                           # [B, 160]
        return self.net(masked)                                    # [B, 784]


class CapsNet(nn.Module):
    """
    Full CapsNet: Conv1 → PrimaryCaps → DigitCaps → Decoder
    """
    def __init__(self, num_routing=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.primary_caps = PrimaryCaps(
            in_channels=256, num_capsules=32, caps_dim=8,
            kernel_size=9, stride=2
        )
        self.digit_caps = DigitCaps(
            in_caps=32 * 6 * 6, in_dim=8,
            num_classes=10, caps_dim=16,
            num_routing=num_routing
        )
        self.decoder = Decoder(caps_dim=16, num_classes=10, img_size=784)

    def forward(self, x, labels=None):
        """
        x:      [B, 1, 28, 28]
        labels: [B] (required during training for decoder mask; at inference
                     the predicted class is used)
        returns:
          v_lengths: [B, 10]  — capsule output vector norms (class probabilities)
          reconstruction: [B, 784]
          v: [B, 10, 16]
        """
        feat = self.conv1(x)                      # [B, 256, 20, 20]
        u = self.primary_caps(feat)               # [B, 1152, 8]
        v = self.digit_caps(u)                    # [B, 10, 16]
        v_lengths = v.norm(dim=-1)                # [B, 10]

        if labels is None:
            labels = v_lengths.argmax(dim=1)

        reconstruction = self.decoder(v, labels)  # [B, 784]
        return v_lengths, reconstruction, v


def margin_loss(v_lengths, targets, num_classes=10, m_plus=0.9, m_minus=0.1, lam=0.5):
    """
    Margin loss (Eq. 4 in paper).
    v_lengths: [B, 10]
    targets:   [B] long
    """
    T = F.one_hot(targets, num_classes=num_classes).float()   # [B, 10]
    loss = T * F.relu(m_plus - v_lengths) ** 2 \
         + lam * (1 - T) * F.relu(v_lengths - m_minus) ** 2
    return loss.sum(dim=1).mean()


def reconstruction_loss(reconstruction, images):
    """MSE between reconstructed and original image pixels."""
    return F.mse_loss(reconstruction, images.view(images.size(0), -1))


def total_loss(v_lengths, targets, reconstruction, images, recon_scale=0.0005):
    m = margin_loss(v_lengths, targets)
    r = reconstruction_loss(reconstruction, images)
    return m + recon_scale * r, m, r
