"""PyTorch Conv-Autoencoder for early anomaly detection.

Trained only on *normal* log-mel spectrograms. At inference time the
per-sample reconstruction error is the anomaly score: abnormal sounds
were never learned, so they reconstruct poorly and score high.

Input shape: ``(batch, 1, n_mels, frames)`` with both spatial dims
divisible by 8 (three stride-2 stages), e.g. 64 x 64.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["ConvAutoencoder", "calibrate_threshold"]


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error, shape ``(batch,)``."""
        self.eval()
        recon = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=tuple(range(1, x.ndim)))


def calibrate_threshold(errors: np.ndarray | torch.Tensor, quantile: float = 0.95) -> float:
    """Set the anomaly threshold from *normal-only* validation errors."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    values = errors.detach().cpu().numpy() if isinstance(errors, torch.Tensor) else np.asarray(errors)
    if values.size == 0:
        raise ValueError("errors must not be empty")
    return float(np.quantile(values, quantile))
