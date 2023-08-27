from importlib import import_module

from typeguard import typechecked

from configs import BaseConfig
from trainers.base_trainer import BaseTrainer


@typechecked
def get_trainer(config: BaseConfig) -> BaseTrainer:
    module = import_module(f'trainers.{config.model}')
    trainer = getattr(module, f"{config.model}_Trainer")(config)
    return trainer
