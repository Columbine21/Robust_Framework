import numpy as np
from torch.utils.data import DataLoader
from configs import *

from .dataset.eval_dataset import TestDataset
from .dataset.train_dataset import Dataset


TASK_TRAINDATA_MAP = {
    'MSA': Dataset,
    'MIR': Dataset,
}

TASK_EVALDATA_MAP = {
    'MSA': TestDataset,
    'MIR': TestDataset,
}
# hyper parameters for models
EVAL_OPT_MAP = {
    # feature level noise.
    "feat_random_drop": np.linspace(0.0, 1.0, 11),
    "feat_structural_drop": np.linspace(0.0, 1.0, 11),
    # raw video level audio noise.
    "rawa_color_w": np.linspace(0.0, 1.0, 11),
    "rawa_bg_park": np.linspace(0.0, 1.0, 11),
    # raw video level visual noise.
    "rawv_gblur": np.linspace(0.0, 1.0, 11),
    "rawv_impulse_value": np.linspace(0.0, 1.0, 11),
}

def MMDataLoader(configs):

    interval = EVAL_OPT_MAP[configs.eval_noise_type]
    # interval = [0.0]

    datasets = {
        'train': TASK_TRAINDATA_MAP[configs.task.upper()](configs, mode='train'),
        'valid': TASK_TRAINDATA_MAP[configs.task.upper()](configs, mode='valid'),
        'test': [TASK_EVALDATA_MAP[configs.task.upper()](configs, noise_intensity=m) for m in interval]
    }

    dataLoader = dict()
    dataLoader['train'] = DataLoader(
        datasets['train'], 
        batch_size=configs.batch_size, 
        num_workers=configs.num_workers, 
        shuffle=True)
    dataLoader['valid'] = DataLoader(
        datasets['valid'], 
        batch_size=configs.batch_size, 
        num_workers=configs.num_workers, 
        shuffle=True)
    dataLoader['test'] = [
        DataLoader(
            datasets['test'][i], 
            batch_size=configs.batch_size, 
            num_workers=configs.num_workers, 
            shuffle=False)
        for i in range(len(datasets['test'])) ]
    
    return dataLoader