from mmengine.registry import Registry

MODELS = Registry(
    "model",
    locations=["v2sflow.models"],
)

SCHEDULERS = Registry(
    "scheduler",
    locations=["v2sflow.schedulers"],
)

DATASETS = Registry(
    "dataset",
    locations=["v2sflow.datasets"],
)
