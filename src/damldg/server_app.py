"""Collect seeded trials without averaging independent models."""

import json
import math
from pathlib import Path
import time
import numpy as np
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import ServerApp
from .config import Config
from .training import digest, run_trial, write_json

app = ServerApp()


def summarize(cfg, records):
    records = sorted(records, key=lambda r: r["trial_id"])
    if [r["trial_id"] for r in records] != list(range(cfg.trials)):
        raise ValueError("Missing or duplicate trial results")
    if any(r["seed"] != cfg.seed + r["trial_id"] for r in records):
        raise ValueError("Trial seed mismatch")
    targets = [r["target"] for r in records]
    if any(t is None for t in targets):
        if not all(t is None for t in targets):
            raise ValueError("Inconsistent trial evaluation")
        return {"config": cfg.as_dict(), "trials": records}
    if len({t["num_examples"] for t in targets}) != 1 or targets[0]["num_examples"] < 1:
        raise ValueError("Trial target sizes disagree")
    for t in targets:
        if not math.isfinite(t["loss"]) or t["loss"] < 0 or not 0 <= t["accuracy"] <= 1:
            raise ValueError("Invalid trial metrics")
    acc = [t["accuracy"] for t in targets]
    return {
        "config": cfg.as_dict(),
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else None,
        "trials": records,
    }


def replies(grid, messages, timeout):
    received = list(grid.send_and_receive(messages, timeout=timeout))
    if len(received) != len(messages):
        raise RuntimeError("Flower trial response timed out")
    for reply in received:
        if reply.has_error():
            raise RuntimeError(f"Flower worker failed: {reply.error}")
    return received


def identify(grid, cfg):
    deadline = time.monotonic() + cfg.registration_timeout
    while time.monotonic() < deadline:
        nodes = list(grid.get_node_ids())
        if len(nodes) == cfg.trials:
            break
        if len(nodes) > cfg.trials:
            raise ValueError("Too many Flower workers")
        time.sleep(0.1)
    else:
        raise TimeoutError("Flower workers did not register")
    messages = [
        Message(
            RecordDict({"config": ConfigRecord({"phase": "identify"})}),
            dst_node_id=n,
            message_type="train",
        )
        for n in nodes
    ]
    mapping = {}
    for reply in replies(grid, messages, cfg.registration_timeout):
        tid = reply.content["metrics"]["trial_id"]
        if type(tid) is not int or not 0 <= tid < cfg.trials or tid in mapping:
            raise ValueError("Invalid or duplicate Flower trial identity")
        mapping[tid] = reply.metadata.src_node_id
    return mapping


def run_experiment(cfg, grid=None):
    Path(cfg.output).mkdir(parents=True, exist_ok=True)
    if grid is None:
        records = [run_trial(cfg, i) for i in range(cfg.trials)]
    else:
        mapping = identify(grid, cfg)
        messages = [
            Message(
                RecordDict(
                    {"config": ConfigRecord({"phase": "train", "trial_id": tid})}
                ),
                dst_node_id=node,
                message_type="train",
                ttl=cfg.reply_timeout,
            )
            for tid, node in mapping.items()
        ]
        records = []
        seen = set()
        for reply in replies(grid, messages, cfg.reply_timeout):
            result = json.loads(reply.content["result"]["json"])
            tid = result["trial_id"]
            if (
                type(tid) is not int
                or tid not in mapping
                or tid in seen
                or mapping[tid] != reply.metadata.src_node_id
            ):
                raise ValueError("Trial response identity mismatch")
            path = Path(cfg.output) / f"seed-{cfg.seed + tid}" / "checkpoint.pt"
            if digest(path) != result["checkpoint_sha256"]:
                raise ValueError("Trial checkpoint checksum mismatch")
            seen.add(tid)
            records.append(result)
    result = summarize(cfg, records)
    result["execution"] = "flower-trials" if grid else "sequential-trials"
    write_json(Path(cfg.output) / "results.json", result)
    return result


@app.main()
def main(grid, context):
    run_experiment(Config(**dict(context.run_config)), grid)
