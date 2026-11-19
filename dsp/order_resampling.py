"""Order resampling: time domain -> order (angle) domain.

Rotating machinery sounds are locked to shaft/wheel rotation, not to
wall-clock time. When RPM drifts, a fault tone smears across Hz bins.
Resampling the signal onto a uniform *rotation phase* grid aligns every
revolution: a component at ``k`` cycles per revolution (order ``k``)
then stays at order ``k`` regardless of speed.

Typical pipeline for one AudioFrame stream::

    phase   = compute_phase(time_s, wheel_rps)          # revolutions
    angled  = order_resample(pcm, phase, samples_per_rev=64)
    orders, magnitude = order_spectrum(angled, samples_per_rev=64)

Only numpy is required.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "speed_kmh_to_wheel_rps",
    "rpm_to_rps",
    "compute_phase",
    "order_resample",
    "order_spectrum",
    "resample_by_speed",
]


def rpm_to_rps(rpm: float) -> float:
    """Revolutions per minute -> revolutions per second."""
    if rpm < 0:
        raise ValueError("rpm must be >= 0")
    return float(rpm) / 60.0


def speed_kmh_to_wheel_rps(speed_kmh: float, tire_radius_m: float = 0.32) -> float:
    """Vehicle speed -> wheel rotation rate (rev/s).

    Assumes no wheel slip: v = omega * r, one revolution covers
    ``2 * pi * tire_radius_m`` metres.
    """
    if speed_kmh < 0:
        raise ValueError("speed_kmh must be >= 0")
    if tire_radius_m <= 0:
        raise ValueError("tire_radius_m must be > 0")
    v_ms = float(speed_kmh) * 1000.0 / 3600.0
    return v_ms / (2.0 * np.pi * tire_radius_m)


def compute_phase(time_s: np.ndarray, rps: np.ndarray | float) -> np.ndarray:
    """Cumulative rotation phase, in revolutions, starting at 0.

    Integrates the instantaneous rotation rate with the trapezoidal
    rule. ``rps`` may be a scalar (constant speed) or one value per
    sample.
    """
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time_s must be a 1-D array with >= 2 samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing")

    rate = np.broadcast_to(np.asarray(rps, dtype=np.float64), t.shape)
    if np.any(rate < 0):
        raise ValueError("rps must be >= 0")

    phase = np.zeros_like(t)
    phase[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))
    return phase


def order_resample(
    signal: np.ndarray,
    phase_rev: np.ndarray,
    samples_per_rev: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ``signal`` onto a uniform phase grid.

    Returns ``(resampled, uniform_phase_rev)`` where ``resampled[i]``
    is the signal at ``i / samples_per_rev`` revolutions. Only complete
    samples inside the recorded phase range are returned.
    """
    x = np.asarray(signal, dtype=np.float64)
    phase = np.asarray(phase_rev, dtype=np.float64)
    if x.shape != phase.shape or x.ndim != 1:
        raise ValueError("signal and phase_rev must be 1-D arrays of equal length")
    if samples_per_rev <= 0:
        raise ValueError("samples_per_rev must be > 0")
    if phase[-1] <= 0:
        raise ValueError("total rotation must be > 0 revolutions")

    n = int(np.floor(phase[-1] * samples_per_rev))
    if n < 2:
        raise ValueError("recording covers less than 2 resampled samples")
    uniform = np.arange(n, dtype=np.float64) / samples_per_rev
    resampled = np.interp(uniform, phase, x)
    return resampled.astype(np.float32), uniform


def order_spectrum(
    resampled: np.ndarray, samples_per_rev: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of an order-resampled signal.

    Returns ``(orders, magnitude)``; ``orders`` are in cycles per
    revolution, so a peak at ``orders == k`` is the k-th order.
    A Hann window is applied to reduce leakage.
    """
    x = np.asarray(resampled, dtype=np.float64)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("resampled must be a 1-D array with >= 4 samples")
    windowed = (x - x.mean()) * np.hanning(x.size)
    spectrum = np.fft.rfft(windowed)
    orders = np.fft.rfftfreq(x.size, d=1.0 / samples_per_rev)
    return orders, np.abs(spectrum)


def resample_by_speed(
    signal: np.ndarray,
    sample_rate_hz: float,
    speed_kmh: np.ndarray | float,
    samples_per_rev: int = 64,
    tire_radius_m: float = 0.32,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: PCM + speed -> order-domain signal.

    ``speed_kmh`` may be a scalar or one value per sample (as carried
    by AudioFrame). Returns the same tuple as :func:`order_resample`.
    """
    x = np.asarray(signal)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("signal must be a 1-D array with >= 2 samples")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    time_s = np.arange(x.size, dtype=np.float64) / float(sample_rate_hz)
    speed = np.broadcast_to(np.asarray(speed_kmh, dtype=np.float64), x.shape)
    rps = np.array(
        [speed_kmh_to_wheel_rps(float(s), tire_radius_m) for s in speed]
    )
    phase = compute_phase(time_s, rps)
    return order_resample(x.astype(np.float64), phase, samples_per_rev)
