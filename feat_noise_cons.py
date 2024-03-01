import pickle
import argparse

from copy import deepcopy
from configs import *
from easydict import EasyDict as edict
from utils.feat_noise import feature_noise

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SIMSv2', choices=['MOSI', 'MOSEI', 'SIMSv2', 'MIntRec'],
                        help='Video Understanding Dataset Name.')
    parser.add_argument('--injected-noise', type=str, default='feat_structural_drop',
                        help='Evaluation Noise Type (validation and test). [feat_structural_drop,feat_random_drop]')
    parser.add_argument('--noise-intensity', type=float, default=0.2,
                        help='Noise Intensity (0.0 ~ 1.0).')
    parser.add_argument('--inject-noise-seed', type=list, default=[1111],
                        help='Random seed for injecting noise.')
    parser.add_argument('--save-dir', type=str, default='/home/sharing/disk3/Datasets/MMSA-Noise',
                        help='Path to save constructed noisy databases for model training or evaluation.')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                        help='Path to save log files. Default: "~/results/logs"')
    
    return parser.parse_args()


if __name__ == '__main__':
    # Load original databases.
    config = edict(**vars(parse_args()))

    feature_origin = Path(DATASET_ROOT_DIR, DATASET_ORI_FEATURE_MAP[config.dataset])
    with open(feature_origin, 'rb') as f:
        data = pickle.load(f)

    # Injecting Feature Level Noise.
    noise_data = deepcopy(data)
    print(f'Injecting Feature Level Noise - {config.injected_noise} - {config.noise_intensity} - {config.inject_noise_seed}...')
    for mode in ['train', 'valid', 'test']:
        noise_data[mode]['text_bert'], noise_data[mode]['audio'], noise_data[mode]['vision'], \
            noise_data[mode]['text_missing_mask'], noise_data[mode]['audio_missing_mask'], noise_data[mode]['vision_missing_mask'] = feature_noise(
            mode=config.injected_noise,
            text=data[mode]['text_bert'],
            audio=data[mode]['audio'],
            vision=data[mode]['vision'],
            audio_lengths=data[mode]['audio_lengths'],
            vision_lengths=data[mode]['vision_lengths'],
            missing_rate=config.noise_intensity, 
            seeds=config.inject_noise_seed
        )
    # Remove Unnecessary Keys.
    # for mode in ['train', 'valid', 'test']:
    #     noise_data[mode].pop('asr_text')
    #     noise_data[mode].pop('asr_bert')
    #     noise_data[mode].pop('text')
    #     noise_data[mode].pop('annotations')
    # Dump noisy databases.
    print('Dumping noisy databases...')
    save_path = Path(config.save_dir, f'{config.dataset}', f'{config.injected_noise}',
                    f'{config.noise_intensity}_{config.inject_noise_seed[0]}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(noise_data, f)

    print('Constructed Noisy Databases saved at', str(save_path))