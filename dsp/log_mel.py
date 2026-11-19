"""Log-Mel spectrogram front-end (numpy only, no librosa dependency)."""

from __future__ import annotations

import numpy as np

__all__ = ["hz_to_mel", "mel_to_hz", "mel_filterbank", "log_mel_spectrogram"]


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate_hz: int = 16000,
    n_fft: int = 512,
    n_mels: int = 64,
    fmin_hz: float = 20.0,
    fmax_hz: float | None = None,
) -> np.ndarray:
    """Triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``."""
    if fmax_hz is None:
        fmax_hz = sample_rate_hz / 2.0
    if not (0 <= fmin_hz < fmax_hz <= sample_rate_hz / 2.0):
        raise ValueError("require 0 <= fmin < fmax <= sample_rate/2")

    mel_points = np.linspace(hz_to_mel(fmin_hz), hz_to_mel(fmax_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate_hz).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def log_mel_spectrogram(
    signal: np.ndarray,
    sample_rate_hz: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute a log-mel spectrogram.

    Input is a 1-D waveform (int16 PCM is normalised to [-1, 1)).
    Output shape is ``(n_mels, n_frames)``, float32, in log10 power.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < n_fft:
        raise ValueError(f"signal must be 1-D with >= {n_fft} samples")
    if np.max(np.abs(x)) > 1.5:  # looks like integer PCM, not float audio
        x = x / 32768.0

    fb = mel_filterbank(sample_rate_hz, n_fft, n_mels)
    window = np.hanning(n_fft)
    n_frames = 1 + (x.size - n_fft) // hop_length
    frames = np.stack(
        [x[i * hop_length : i * hop_length + n_fft] * window for i in range(n_frames)]
    )
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (frames, bins)
    mel_power = power @ fb.T                          # (frames, n_mels)
    return np.log10(mel_power + eps).T.astype(np.float32)  # (n_mels, frames)
