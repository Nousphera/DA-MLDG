"""Flower workers execute independent complete source-domain trials."""

import json
from flwr.app import ConfigRecord, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .config import Config

app = ClientApp()


@app.train()
def train(message, context):
    cfg = Config(**dict(context.run_config))
    trial_id = context.node_config["partition-id"]
    if type(trial_id) is not int or not 0 <= trial_id < cfg.trials:
        raise ValueError("Invalid trial partition")
    phase = message.content["config"]["phase"]
    if phase == "identify":
        return Message(
            RecordDict({"metrics": MetricRecord({"trial_id": trial_id})}),
            reply_to=message,
        )
    if phase != "train" or message.content["config"]["trial_id"] != trial_id:
        raise ValueError("Trial assignment does not match this worker")
    from .training import run_trial

    result = run_trial(cfg, trial_id)
    return Message(
        RecordDict(
            {"result": ConfigRecord({"json": json.dumps(result, allow_nan=False)})}
        ),
        reply_to=message,
    )
