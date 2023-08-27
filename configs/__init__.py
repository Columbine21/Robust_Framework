from importlib import import_module

from typeguard import typechecked

from configs.base_config import *


@typechecked
def get_config(model: ALL_MODELS_LITERAL, dataset: ALL_DATASETS_LITERAL, **kwargs) -> BaseConfig:
    module = import_module(f'configs.{model}')
    config = getattr(module, f"{model}_Config")(model=model, dataset=dataset, **kwargs)
    return config