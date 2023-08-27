import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging

from utils.metrics import MIRmetrics, MSAmetrics
from utils.functions import AIRrobustness

TASK_METRICS_MAP = {
    'MSA': MSAmetrics,
    'MIR': MIRmetrics,   
}

class BaseTrainer(object):
    def __init__(self, config) -> None:
        self.config = config
        self.criterion = nn.L1Loss() if config.task == 'MSA' else nn.CrossEntropyLoss()
        self.metrics = TASK_METRICS_MAP[config.task]().getMetics(config.dataset)
        self.config.KeyEval = 'MAE' if config.task == 'MSA' else 'Loss'
        # Initilization
        self.epochs, self.best_epoch, self.best_valid = 0, 0, 1e8
        self.logger = logging.getLogger('OpenVNA')

    def do_train(self, model, dataloader):
        NotImplementedError

    def do_valid(self, model, dataloader):
        NotImplementedError

    def do_robustness_test(self, model, dataloaders):
        results = {}

        for n, dataloader in enumerate(dataloaders):            
            result_ = self.do_valid(model, dataloader, mode=f"Robustness Test {n}")
            results[n] = result_

        if len(dataloaders) > 1:
            # Using Extended Arbitrary Interval Robustness Metrics.
            result_int = dict()
            for k in list(results[list(results.keys())[0]].keys()):
                result_int[k] = AIRrobustness([results[v][k] for v in list(results.keys())])[0]

        return results if len(dataloaders) == 1 else result_int
