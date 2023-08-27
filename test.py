import os, gc
import logging
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from utils.functions import setup_seed, count_parameters
from configs import *
from models import get_model
from trainers import get_trainer
from dataloader.dataloader import MMDataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def set_logger(config: BaseConfig):
    log_file_path = Path(config.log_dir, f'{config.model}-{config.augmentation}-{config.dataset}.log')
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    # set logging
    logger = logging.getLogger('OpenVNA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[config.verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='TPFN', choices=['T2FN', 'TPFN', 'CTFN', 'MMIN', 'TFRNet', 'GCNET', 'NIAT', 'EMT_DLFR'],
                        help='Robust Baselines Name.')
    parser.add_argument('--dataset', type=str, default='MOSI', choices=['MOSI', 'MOSEI', 'SIMSv2', 'MIntRec'],
                        help='Video Understanding Dataset Name.')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers for data loader.')
    parser.add_argument('--eval-noise-type', type=str, default='feat_random_drop',
                        help='Evaluation Noise Type (validation and test).')
    parser.add_argument('--test-missing-seed', type=list, default=[1111],
                        help='Test Missing Seed. (used in feature level noisy instance construction)')
    parser.add_argument('--seeds', type=list, default=[0, 1, 2],
                        help='Seeds for training.')
    parser.add_argument('--device', action='append', default=[1],
                        help='Specify which gpus to use. If an empty list is supplied, will automatically assign to the most memory-free gpu. \
                            Currently only support single gpu. Default: []')
    parser.add_argument('--verbose_level', type=int, default=1,
                        help='Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model-save-dir', type=str, default='results/saved_models',
                        help='Path to save trained models. Default: "~/results/saved_models"')
    parser.add_argument('--res-save-dir', type=str, default='results/results',
                        help='Path to save csv results. Default: "~/results/results"')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                        help='Path to save log files. Default: "~/results/logs"')
    
    return parser.parse_args()

def _run(config,dataloader):
    
    model = get_model(config).to(config.device)
    logger.info(f'The model has {count_parameters(model)} trainable parameters.')
    trainer = get_trainer(config)
    # load trained model & do test
    assert Path(config.model_save_path).exists()
    model.load_state_dict(torch.load(config.model_save_path))
    model.to(config.device)
    results = trainer.do_robustness_test(model, dataloader)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

if __name__ == "__main__":

    config = get_config(**vars(parse_args()))
    # set logger.
    config.log_dir = Path(config.log_dir, f'{config.task}-{config.dataset}')
    config.log_dir.mkdir(parents=True, exist_ok=True)
    logger = set_logger(config=config)
    # set results save dir.
    config.res_save_dir = Path(config.res_save_dir, f'{config.dataset}')
    config.res_save_dir.mkdir(parents=True, exist_ok=True)
    # set original, augmented feature path.
    config.task = DATASET_TASK_MAP[config.dataset]
    config.feature_origin = Path(DATASET_ROOT_DIR, DATASET_ORI_FEATURE_MAP[config.dataset])
    config.feature_augmented = []
    config.augmentation = [] if config.augmentation == None else config.augmentation
    for n_t in config.augmentation:
        noise_dir = Path(NOISY_DATASET_ROOT_DIR, f'{config.dataset}', f'{n_t}')
        files = [Path(noise_dir, fn) for fn in os.listdir(noise_dir)]
        config.feature_augmented.extend(files)
    # Program Start.
    logger.info("======================================== Program Start ========================================")
    torch.cuda.set_device(config.device)
    logger.info("Running with args:")
    logger.info(config)
    logger.info(f"Seeds: {config.seeds}")

    model_results = []
    dataloader = MMDataLoader(config)
    dataloader = dataloader['test']

    for i, seed in enumerate(config.seeds):
        setup_seed(seed)
        config.model_save_path = Path(config.model_save_dir, f'{config.dataset}',
                                    f'{config.model}-{config.augmentation}-{seed}.pth')
        config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        config.cur_seed = i + 1
        logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(config.seeds)}] {'-'*30}")
        # actual running
        result = _run(config, dataloader)
        logger.info(f"Result for seed {seed}: {result}")
        model_results.append(result)

    criterions = list(model_results[0].keys())
    # save result to csv
    csv_file = Path(config.res_save_dir, f"{config.eval_noise_type}.csv")
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['Model'] + ['Augmentation'] + criterions)
    # save results
    res = [config.model, config.augmentation]
    for c in criterions:
        values = [r[c] for r in model_results]
        res.append((round(np.mean(values)*100, 2), round(np.std(values)*100, 2)))
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}.")
