import logging
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import DATASET_ROOT_DIR, BaseConfig, NOISY_DATASET_ROOT_DIR
from utils.feat_noise import feature_noise

class TestDataset(Dataset):
    def __init__(self, config: BaseConfig, noise_intensity: float = 0.0) -> None:
        self.config = config
        self.logger = logging.getLogger("OpenVNA")
        self.mode = 'test'
        
        if "feat" in config.eval_noise_type:
            # Load Original Feature.
            assert Path(DATASET_ROOT_DIR, self.config.feature_origin).is_file(), \
                f"Feature file {Path(DATASET_ROOT_DIR, self.config.feature_origin)} does not exist."
            text, audio, vision, self.labels, self.ids, self.audio_lengths, self.vision_lengths = \
                self._load_feature(config.feature_origin)
            # Feature level noise.
            self.text, self.audio, self.vision, _, _, _= feature_noise(
                                                            mode=config.eval_noise_type, 
                                                            text=text,
                                                            audio=audio, 
                                                            vision=vision, 
                                                            audio_lengths=self.audio_lengths, 
                                                            vision_lengths=self.vision_lengths, 
                                                            missing_rate=noise_intensity, 
                                                            seeds=config.test_missing_seed
                                                        )
        elif "raw" in config.eval_noise_type:
            noise_intensity = round(noise_intensity,1)
            if noise_intensity == 0.0:
                file_path = Path(DATASET_ROOT_DIR, self.config.feature_origin)
            else:
                file_path = Path(NOISY_DATASET_ROOT_DIR,self.config.dataset,'noise_test', self.config.eval_noise_type,f"miss_{noise_intensity}.pkl")
            # Raw Video Level noise.
            assert file_path.is_file(), f"Feature file {file_path} does not exist."
            
            self.text, self.audio, self.vision, self.labels,\
                self.ids, self.audio_lengths, self.vision_lengths = self._load_feature(file_path)
            
        self.logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
            
    def _load_feature(self, feature_file: str):
        self.logger.debug(f"Loading feature from {feature_file}")
        with open(feature_file, "rb") as f:
            data = pickle.load(f)
        # Load Text Features.
        text = data[self.mode]["text_bert"].astype(np.float32)
        # Load Audio Features.
        audio = data[self.mode]["audio"].astype(np.float32)
        audio[audio == -np.inf] = 0
        audio_lengths = data[self.mode]["audio_lengths"] if "audio_lengths" in data[self.mode] else np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
        # Load Vision Features.
        vision = data[self.mode]["vision"].astype(np.float32)
        vision_lengths = data[self.mode]["vision_lengths"] if "vision_lengths" in data[self.mode] else np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
        # Load Labels.
        if self.config.task == 'MIR':
            labels = {'M': data[self.mode]["labels"].astype(np.float32)}
        else:
            labels = {'M': data[self.mode]["regression_labels"].astype(np.float32)}
        # if self.config.dataset in ["SIMS", "SIMSv2"]:
        #     for m in ["T", "A", "V"]:
        #         labels[m] = data[self.mode][f"regression_labels_{m}"].astype(np.float32)
        ids = data[self.mode]["id"]

        return text, audio, vision, labels, ids, audio_lengths, vision_lengths

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'audio_lengths': self.audio_lengths[index],
            'vision_lengths': self.vision_lengths[index],
        }
        
        return sample