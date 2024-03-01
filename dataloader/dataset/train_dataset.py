import logging
import pickle
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from configs import *

class Dataset(Dataset):
    def __init__(self, config: BaseConfig, mode: Literal["train", "valid"]):
        self.config = config
        self.logger = logging.getLogger("OpenVNA")
        self.mode = mode
        self.config.feature_dims = [768, 0, 0]
        self.config.seq_lens = [0, 0, 0]
        # Load Original Feature.
        assert Path(self.config.feature_origin).is_file(), \
            f"Feature file {Path(self.config.feature_origin)} does not exist."
        text, audio, vision, self.labels, self.ids, self.audio_lengths, self.vision_lengths = \
            self._load_feature(config.feature_origin)


        self.config.seq_lens[0], self.config.seq_lens[1], self.config.seq_lens[2] = \
            text.shape[2], audio.shape[1], vision.shape[1]
        self.config.feature_dims[1], self.config.feature_dims[2] = audio.shape[2], vision.shape[2]
        self.text_l, self.audio_l, self.vision_l = {}, {}, {}
        # Constructing Optional Feature List.
        for n, index in enumerate(self.ids):
            self.text_l[index] = [('origin', text[n])]
            self.audio_l[index] = [('origin', audio[n])]
            self.vision_l[index] = [('origin', vision[n])]

        # Load Raw Video Level Noise-based Augmented Feature.
        if self.config.feature_augmented is not None:
            # Augmentation Format -> {ids: [ (opt, [text/audio/vision]) , ...]}
            for aug_path in self.config.feature_augmented:
                aug_opt = str(aug_path).split("/")[-2] + '_' + str(aug_path).split("/")[-1].replace('.pkl', '') # e.g. "feat_random_drop_0.5_1111"
                assert Path(aug_path).is_file(), f"Feature file {aug_path} does not exist."
                text_, audio_, vision_, _, ids_, _, _ = self._load_feature(aug_path)
                for n, index in enumerate(ids_):
                    if not index in self.ids:
                        continue
                    self.text_l[index].append((aug_opt, text_[n]))
                    self.audio_l[index].append((aug_opt, audio_[n]))
                    self.vision_l[index].append((aug_opt, vision_[n]))
        if self.config.coupled_instance:
            assert len(self.config.feature_augmented) > 0, \
                "Coupled Instance Training requires at least one augmentation."

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
        audio_lengths = data[self.mode]["audio_lengths"] if "audio_lengths" in data[self.mode] \
            else np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
        # Load Vision Features.
        vision = data[self.mode]["vision"].astype(np.float32)
        vision_lengths = data[self.mode]["vision_lengths"] if "vision_lengths" in data[self.mode] \
            else np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
        # Load Labels.
        if self.config.task == 'MIR':
            labels = {'M': data[self.mode]["labels"].astype(np.float32)}
        else:
            labels = {'M': data[self.mode]["regression_labels"].astype(np.float32)}
        # if self.config.dataset in ["SIMS", "SIMSv2"]:
        #     for m in ["T", "A", "V"]:
        #         labels[m] = data[self.mode][f"regression_labels_{m}"].astype(np.float32)
        # raw_text = data[self.mode]["raw_text"]
        ids = data[self.mode]["id"]

        return text, audio, vision, labels, ids, audio_lengths, vision_lengths

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        vid = self.ids[index]
        
        sample = {
            'text': torch.Tensor(random.choice(self.text_l[vid])[1]),
            'audio': torch.Tensor(random.choice(self.audio_l[vid])[1]),
            'vision': torch.Tensor(random.choice(self.vision_l[vid])[1]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'audio_lengths': self.audio_lengths[index],
            'vision_lengths': self.vision_lengths[index],
        }
        if self.config.coupled_instance:
            sample.update( {
                'text': torch.Tensor(self.text_l[vid][0][1]), # Using original feature.
                'audio': torch.Tensor(self.audio_l[vid][0][1]),
                'vision': torch.Tensor(self.vision_l[vid][0][1]),
                'text_m': torch.Tensor(random.choice(self.text_l[vid][1:])[1]), # Using augmented feature.
                'audio_m': torch.Tensor(random.choice(self.audio_l[vid][1:])[1]),
                'vision_m': torch.Tensor(random.choice(self.vision_l[vid][1:])[1]),
            } )

        return sample
