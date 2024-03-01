import numpy as np
import pandas as pd
import altair as alt
import torch
import torch.nn as nn
import logging
from altair_saver import save

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
        record_block_mosi_rate_miss = {
            'Model': [self.config.model] * 11,
            'Missing Rate': [0.0, 0.1 , 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            'Accuracy':[]
        }
        for n, dataloader in enumerate(dataloaders):            
            result_ = self.do_valid(model, dataloader, mode=f"Robustness Test {n}")
            record_block_mosi_rate_miss['Accuracy'].append(result_['Non0_acc_2']*100)
            results[n] = result_
        
        if self.config.chart:
            record_block_mosi_rate_miss = pd.DataFrame(record_block_mosi_rate_miss)
            chart = alt.Chart(record_block_mosi_rate_miss).mark_line(
                point=True
            ).encode(
                x='Missing Rate',
                y=alt.Y('Accuracy', scale=alt.Scale(domain=[40,90])),
                color='Model',
                strokeDash='Model',
            ).properties(height=180, width=290)
            chart.save(f'{self.config.res_save_dir}/{self.config.model}_{self.config.cur_seed-1}_chart.png')

        if len(dataloaders) > 1:
            # Using Extended Arbitrary Interval Robustness Metrics.
            result_int = dict()
            for k in list(results[list(results.keys())[0]].keys()):
                result_int[k] = AIRrobustness([results[v][k] for v in list(results.keys())])[0]

        return results if len(dataloaders) == 1 else result_int
