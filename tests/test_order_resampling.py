import numpy as np
import pytest

from dsp.order_resampling import (
    compute_phase,
    order_resample,
    order_spectrum,
    resample_by_speed,
    rpm_to_rps,
    speed_kmh_to_wheel_rps,
)


def test_speed_conversion():
    # 36 km/h = 10 m/s; circumference 2*pi*0.32 ~= 2.01 m -> ~4.97 rev/s
    assert speed_kmh_to_wheel_rps(36.0, tire_radius_m=0.32) == pytest.approx(4.97, abs=0.01)
    assert rpm_to_rps(60.0) == 1.0
    with pytest.raises(ValueError):
        speed_kmh_to_wheel_rps(-1.0)


def test_constant_speed_tone_lands_on_its_order():
    sample_rate = 16000
    duration_s = 1.0
    rps = 10.0            # constant rotation rate
    order = 3.0           # tone at 3 cycles per revolution -> 30 Hz
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = np.sin(2 * np.pi * order * rps * t)

    phase = compute_phase(t, np.full_like(t, rps))
    assert phase[-1] == pytest.approx(rps * duration_s, rel=1e-3)

    resampled, uniform = order_resample(signal, phase, samples_per_rev=64)
    assert resampled.shape == uniform.shape
    assert len(resampled) == pytest.approx(10 * 64, abs=2)

    orders, magnitude = order_spectrum(resampled, samples_per_rev=64)
    peak_order = orders[np.argmax(magnitude[1:]) + 1]
    assert peak_order == pytest.approx(order, abs=0.2)


def test_resample_by_speed_scalar_and_array_agree():
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    signal = np.sin(2 * np.pi * 50 * t)
    a, _ = resample_by_speed(signal, sample_rate, 36.0)
    b, _ = resample_by_speed(signal, sample_rate, np.full_like(t, 36.0))
    np.testing.assert_allclose(a, b)


def test_validation():
    with pytest.raises(ValueError):
        compute_phase(np.array([0.0, 0.0]), 1.0)  # not increasing
    with pytest.raises(ValueError):
        order_resample(np.zeros(10), np.zeros(10))  # no rotation
