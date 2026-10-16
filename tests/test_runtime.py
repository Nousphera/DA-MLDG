import json
import numpy as np
import pytest
from damldg.config import Config


@pytest.mark.parametrize(
    "override",
    [
        {"trials": 0},
        {"epochs": 0},
        {"learning_rate": float("nan")},
        {"momentum": 1.0},
        {"initialization": "bad"},
        {"backend": "flower", "device": "cuda:1"},
    ],
)
def test_invalid_config(override):
    with pytest.raises(ValueError):
        Config(**override)


def test_summary_requires_every_trial():
    from damldg.server_app import summarize

    cfg = Config(trials=2)
    records = [
        {
            "trial_id": 0,
            "seed": 2021,
            "target": {"accuracy": 0.25, "loss": 1.5, "num_examples": 10},
        }
    ]
    with pytest.raises(ValueError):
        summarize(cfg, records)
    records.append(
        {
            "trial_id": 1,
            "seed": 2022,
            "target": {"accuracy": 0.75, "loss": 0.5, "num_examples": 10},
        }
    )
    result = summarize(cfg, records)
    assert result["accuracy_mean"] == 0.5
    assert result["accuracy_std"] == pytest.approx(np.std([0.25, 0.75], ddof=1))


def test_flower_descriptor_is_temporary():
    from damldg.experiment import flower_app

    with flower_app(Config().as_dict()) as path:
        assert (path / "pyproject.toml").is_file()
        assert (path / "damldg/config.py").is_file()
        saved = path
    assert not saved.exists()


def test_flower_identify(monkeypatch):
    from flwr.supercore.task_identity import TaskIdentity

    for key in ("_run_id", "_task_id", "_node_id"):
        monkeypatch.setattr(TaskIdentity, key, 1)
    from flwr.app import Context, ConfigRecord, Message, RecordDict
    from damldg.client_app import train

    cfg = Config(trials=2).as_dict()
    context = Context(
        run_id=1,
        node_id=10,
        node_config={"partition-id": 1},
        state=RecordDict(),
        run_config=cfg,
    )
    incoming = Message(
        RecordDict({"config": ConfigRecord({"phase": "identify"})}),
        dst_node_id=10,
        message_type="train",
    )
    reply = train(incoming, context)
    assert reply.content["metrics"]["trial_id"] == 1


def make_manifest(tmp_path, representation="features", shape=(1, 64, 64)):
    rows = []
    for domain, split in [("a", "source"), ("b", "source"), ("c", "target")]:
        for index in range(2):
            path = tmp_path / f"{domain}-{index}.npy"
            np.save(
                path, np.random.default_rng(index).normal(size=shape).astype("float32")
            )
            rows.append(
                {
                    "id": path.stem,
                    "path": str(path),
                    "label": index,
                    "domain": domain,
                    "split": split,
                }
            )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "classes": ["zero", "one"],
                "representation": representation,
                "records": rows,
            }
        )
    )
    return path


def test_reject_singleton_embedding_rsc(tmp_path):
    from damldg.training import prepare_run

    manifest = make_manifest(tmp_path, shape=(1, 32, 32))
    cfg = Config(manifest=str(manifest), method="rsc", initialization="random")
    with pytest.raises(ValueError, match="RSC"):
        prepare_run(cfg)


def test_reject_singleton_meta_batch(tmp_path):
    from damldg.training import prepare_run

    manifest = make_manifest(tmp_path, shape=(1, 32, 32))
    cfg = Config(manifest=str(manifest), batch_size=2, initialization="random")
    with pytest.raises(ValueError, match="batch"):
        prepare_run(cfg)


@pytest.mark.parametrize("method", ["deepall", "mldg", "damldg", "rsc"])
def test_complete_trial_checkpoint_reload(tmp_path, method):
    from damldg.training import run_trial, load_checkpoint, evaluate
    from damldg.data import load_manifest

    manifest = make_manifest(tmp_path)
    cfg = Config(
        manifest=str(manifest),
        output=str(tmp_path / "results"),
        method=method,
        initialization="random",
        trials=1,
        epochs=1,
        batch_size=8,
        backend="sequential",
        learning_rate=0.001,
    )
    metrics = run_trial(cfg, 0)
    model, saved = load_checkpoint(tmp_path / "results/seed-2021/checkpoint.pt")
    rows = [r for r in load_manifest(manifest)["records"] if r["split"] == "target"]
    assert evaluate(model, rows, "features") == metrics["target"]
    assert saved["provenance"]["manifest_sha256"]
    assert metrics["source_examples"] == 4
    assert len(saved["history"]) == 1


def test_local_service_uses_distinct_free_ports():
    from damldg.runtime import available_ports
    import socket

    first, second = available_ports()
    assert first != second
    for port in (first, second):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def test_source_bn_calibration_restores_statistics_without_changing_weights(tmp_path):
    import torch
    from torch import nn
    from damldg.training import calibrate_source_bn

    model = nn.Sequential(
        nn.BatchNorm2d(1, momentum=0.01), nn.Flatten(), nn.Linear(4, 2)
    )
    model.train()
    model[0].eval()
    model[0].running_mean.fill_(500)
    model[0].running_var.fill_(900)
    params = {name: value.detach().clone() for name, value in model.named_parameters()}
    modes = [module.training for module in model.modules()]
    records = []
    for index, domain in enumerate(("a", "b")):
        path = tmp_path / f"{domain}.npy"
        np.save(path, np.full((1, 2, 2), 2 + index * 2, dtype="float32"))
        records.append({"path": str(path), "domain": domain, "split": "source"})
    result = calibrate_source_bn(model, records, "features", 2, 2021, "cpu")
    assert result["batches"] == 1
    assert result["source_examples"] == 2
    assert model[0].running_mean.item() == pytest.approx(3)
    assert model[0].running_var.item() == pytest.approx(8 / 7)
    assert model[0].momentum == 0.01
    assert [module.training for module in model.modules()] == modes
    assert all(
        torch.equal(value, params[name]) for name, value in model.named_parameters()
    )
    before = {name: value.clone() for name, value in model.named_buffers()}
    with pytest.raises(ValueError, match="target"):
        calibrate_source_bn(
            model,
            [records[0], {**records[1], "split": "target"}],
            "features",
            2,
            2021,
            "cpu",
        )
    assert all(
        torch.equal(value, before[name]) for name, value in model.named_buffers()
    )
