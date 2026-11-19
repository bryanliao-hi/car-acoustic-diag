import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.conv_autoencoder import ConvAutoencoder, calibrate_threshold


def test_forward_preserves_shape_and_scores():
    model = ConvAutoencoder()
    x = torch.randn(2, 1, 64, 64)
    recon = model(x)
    assert recon.shape == x.shape

    errors = model.reconstruction_error(x)
    assert errors.shape == (2,)
    assert torch.all(errors >= 0)


def test_training_step_reduces_or_runs():
    model = ConvAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 64, 64)
    loss = torch.mean((model(x) - x) ** 2)
    loss.backward()
    opt.step()
    assert np.isfinite(loss.item())


def test_calibrate_threshold():
    errors = np.linspace(0.0, 1.0, 101)
    assert calibrate_threshold(errors, quantile=0.95) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        calibrate_threshold(errors, quantile=1.5)
