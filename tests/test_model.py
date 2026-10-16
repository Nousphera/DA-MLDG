import copy
import pytest
import torch
from torch import nn
from damldg.model import build_model, save_encoder, load_encoder


def test_real_resnet18_architecture_and_feature_head_contract():
    torch.set_num_threads(2)
    model = build_model(num_classes=5, in_channels=1)
    assert [len(getattr(model.encoder, f"layer{i}")) for i in range(1, 5)] == [
        2,
        2,
        2,
        2,
    ]
    assert model.encoder.conv1.kernel_size == (7, 7)
    assert model.encoder.conv1.stride == (2, 2)
    assert sum(p.numel() for p in model.parameters()) > 11_000_000
    assert all(not m.inplace for m in model.modules() if isinstance(m, nn.ReLU))
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    z = model.forward_features(x)
    assert z.shape == (2, 512, 2, 2)
    torch.testing.assert_close(model(x), model.classify(z))
    assert (
        build_model(num_classes=7, in_channels=3, rsc=True).head[-1].out_features == 7
    )


def test_strict_encoder_transfer_ignores_classifier(tmp_path):
    source = build_model(num_classes=11)
    target = build_model(num_classes=5, rsc=True)
    path = tmp_path / "encoder.pt"
    save_encoder(source, path)
    head = copy.deepcopy(target.head.state_dict())
    load_encoder(target, path)
    for k, v in source.encoder.state_dict().items():
        torch.testing.assert_close(target.encoder.state_dict()[k], v, rtol=0, atol=0)
    for k, v in head.items():
        torch.testing.assert_close(target.head.state_dict()[k], v, rtol=0, atol=0)
    loaded = build_model(num_classes=5, encoder_path=path)
    for k, v in source.encoder.state_dict().items():
        torch.testing.assert_close(loaded.encoder.state_dict()[k], v, rtol=0, atol=0)
    with pytest.raises(ValueError, match="channel"):
        build_model(in_channels=3, encoder_path=path)


@pytest.mark.parametrize("kind", ["missing", "shape", "full_model"])
def test_reject_malformed_encoder_without_partial_write(tmp_path, kind):
    model = build_model()
    path = tmp_path / "encoder.pt"
    save_encoder(model, path)
    checkpoint = torch.load(path, weights_only=True)
    if kind == "missing":
        checkpoint["state_dict"].pop("conv1.weight")
    elif kind == "shape":
        checkpoint["state_dict"]["conv1.weight"] = torch.ones(1)
    else:
        checkpoint = {"state_dict": model.state_dict()}
    torch.save(checkpoint, path)
    before = copy.deepcopy(model.encoder.state_dict())
    with pytest.raises(ValueError):
        load_encoder(model, path)
    for k, v in before.items():
        torch.testing.assert_close(model.encoder.state_dict()[k], v, rtol=0, atol=0)
