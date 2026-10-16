import pytest
import torch


def test_trillsson_frontend_shape_and_finiteness():
    from damldg.trillsson import TrillssonFrontend

    x = TrillssonFrontend()(torch.zeros(32000))
    assert x.shape == (1, 1, 1, 195, 80)
    assert torch.isfinite(x).all()


def test_embedding_layout_is_explicit():
    from damldg.weights import embedding_image

    assert embedding_image(torch.arange(1024).float()).shape == (1, 32, 32)
    with pytest.raises(ValueError):
        embedding_image(torch.ones(512))
    with pytest.raises(ValueError):
        embedding_image(torch.full((1024,), float("nan")))


def test_weights_download_rejects_wrong_hash(tmp_path):
    from damldg.weights import download_verified

    source = tmp_path / "source.pt"
    source.write_bytes(b"invalid")
    target = tmp_path / "target.pt"
    with pytest.raises(ValueError):
        download_verified(source.as_uri(), target, "0" * 64)
    assert not target.exists()
