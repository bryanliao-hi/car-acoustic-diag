import numpy as np
import pytest

from dsp.log_mel import hz_to_mel, log_mel_spectrogram, mel_filterbank, mel_to_hz


def test_mel_roundtrip():
    hz = np.array([20.0, 440.0, 1000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, rtol=1e-6)


def test_filterbank_shape_and_nonnegative():
    fb = mel_filterbank(sample_rate_hz=16000, n_fft=512, n_mels=64)
    assert fb.shape == (64, 257)
    assert np.all(fb >= 0)
    assert np.all(fb.max(axis=1) > 0)


def test_log_mel_shape_and_tone_beats_silence():
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
    silence = np.zeros(sr, dtype=np.int16)

    feat_tone = log_mel_spectrogram(tone, sample_rate_hz=sr)
    feat_silence = log_mel_spectrogram(silence, sample_rate_hz=sr)

    assert feat_tone.shape[0] == 64
    assert feat_tone.shape == feat_silence.shape
    assert np.all(np.isfinite(feat_tone))
    assert feat_tone.mean() > feat_silence.mean()


def test_too_short_signal_rejected():
    with pytest.raises(ValueError):
        log_mel_spectrogram(np.zeros(100, dtype=np.int16))
