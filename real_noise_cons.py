from utils.real_noise import real_noise
import pickle
from glob import glob
import time
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from MSA_FET import FeatureExtractionTool
import librosa
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor)
import torch
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from utils.functions import execute_cmd,audio_pad,text_pad,vision_pad
DEVICE = "cuda"
WAV2VEC_MODEL_NAME = "wav2vec2-large-xlsr-53-english"
# WAV2VEC_MODEL_NAME = 'wav2vec2-large-xlsr-53-chinese-zh-cn'
def do_asr(audio_file) -> str:
    try:
        sample_rate = 16000
        speech, _ = librosa.load(audio_file, sr=sample_rate)
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_NAME).to(DEVICE)
        features = processor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="longest"
        )
        with torch.no_grad():
            logits = model(features.input_values.to(DEVICE)).logits.cpu()[0]
        predicted_ids = torch.argmax(logits, dim=-1)
        asr_text = processor.decode(predicted_ids)
        return asr_text
    except Exception as e:
        raise e

def get_asr_text(video_item_path,audio_type):
    audio_save_path = f"assets/temp/audio_{audio_type}.wav"
    cmd = f"ffmpeg -i {video_item_path} -vn -acodec pcm_s16le -ac 1 -y {audio_save_path}"
    execute_cmd(cmd)
    transcript = do_asr(audio_save_path)
    os.remove(audio_save_path)
    return transcript

def audio_color_w_nosie():
    opensmile_fet = FeatureExtractionTool(config='configs/extraction/opensmile.json')
    bert_fet = FeatureExtractionTool(config="configs/extraction/bert_cn.json")
    temp_mp4 = f'assets/temp/audio_color_w.mp4'
    raw_dirs = f"{VIDEO_PATH}/valid_raw/*.mp4"
    for item in tqdm(glob(raw_dirs)):
        name = Path(item).stem
        for n_r in np.arange(0.02, 0.03, 0.01):
            n_r = round(n_r, 2)
            real_noise(
                item,
                temp_mp4,
                mode="percent",
                v_mode=[],
                v_start=[],
                v_end=[],
                v_option=[],
                a_mode=["coloran"],
                a_start=[0.0],
                a_end=[1.0],
                a_option=[("white", n_r)],
            )
            transcript = get_asr_text(temp_mp4,'color_w')
            opensmile_item = opensmile_fet.run_single(temp_mp4)
            os.remove(temp_mp4)
            opensmile_item = audio_pad(opensmile_item)
            if transcript == '':
                transcript = 'pad'
            bert_item = bert_fet.run_single('',text = transcript)
            bert_item = text_pad(bert_item)
            opensmile_item.update(bert_item)
            n_r = round(n_r * 10, 1)
            pickle.dump(opensmile_item, open(f"{SAVE_PATH}/{name}_colorw_{n_r}.pkl",'wb'))
            
def audio_bg_park_noise():
    opensmile_fet = FeatureExtractionTool(config='configs/extraction/opensmile.json')
    bert_fet = FeatureExtractionTool(config="configs/extraction/bert_cn.json")
    temp_mp4 = f'assets/temp/audio_bg_park.mp4'
    raw_dirs = f"{VIDEO_PATH}/valid_raw/*.mp4"
    for item in tqdm(glob(raw_dirs)):
        name = Path(item).stem
        for n_r in np.arange(0.2, 0.3, 0.1):
            n_r = round(n_r, 1)
            real_noise(
                item,
                temp_mp4,
                mode="percent",
                v_mode=[],
                v_start=[],
                v_end=[],
                v_option=[],
                a_mode=["background"],
                a_start=[0.0],
                a_end=[1.0],
                a_option=[("park", n_r)],
            )
            transcript = get_asr_text(temp_mp4,'bg_park')
            opensmile_item = opensmile_fet.run_single(temp_mp4)
            os.remove(temp_mp4)
            opensmile_item = audio_pad(opensmile_item)
            if transcript == '':
                transcript = 'pad'
            bert_item = bert_fet.run_single('',text = transcript)
            bert_item = text_pad(bert_item)
            opensmile_item.update(bert_item)
            pickle.dump(opensmile_item, open(f"{SAVE_PATH}/{name}_bg_park_{n_r}.pkl",'wb'))

def video_gblur_noise():
    openface_fet = FeatureExtractionTool(config='configs/extraction/openface.json')
    temp_mp4 = f'assets/temp/video_gblur.mp4'
    raw_dirs = f"{VIDEO_PATH}/*/*.mp4"
    for item in tqdm(glob(raw_dirs)):
        name = Path(item).stem
        for n_r in np.arange(1, 11, 1):
            real_noise(
                item,
                temp_mp4,
                mode="percent",
                v_mode=["gblur"],
                v_start=[0.0],
                v_end=[1.0],
                v_option=[n_r],
                a_mode=[],
                a_start=[],
                a_end=[],
                a_option=[],
            )
            openface_item = openface_fet.run_single(temp_mp4)
            os.remove(temp_mp4)
            openface_item = vision_pad(openface_item)
            n_r = round((n_r / 10), 1)
            pickle.dump(openface_item, open(f"{SAVE_PATH}/{name}_gblur_{n_r}.pkl",'wb'))

def video_impulse_value_noise():
    openface_fet = FeatureExtractionTool(config='configs/extraction/openface.json')
    temp_mp4 = f'assets/temp/video_impulse.mp4'
    raw_dirs = f"{VIDEO_PATH}/valid_raw/*.mp4"
    for item in tqdm(glob(raw_dirs)):
        name = Path(item).stem
        for n_r in np.arange(20, 30, 10):
            real_noise(
                item,
                temp_mp4,
                mode="percent",
                v_mode=["impulse_value"],
                v_start=[0.0],
                v_end=[1.0],
                v_option=[n_r],
                a_mode=[],
                a_start=[],
                a_end=[],
                a_option=[],
            )
            openface_item = openface_fet.run_single(temp_mp4)
            os.remove(temp_mp4)
            openface_item = vision_pad(openface_item)
            n_r = round((n_r / 100), 1)
            pickle.dump(openface_item, open(f"{SAVE_PATH}/{name}_impulse_{n_r}.pkl",'wb'))
            time.sleep(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, default='/home/sharing/disk2/zhangbaozheng/dataset/simsv2/RAW',
                        help='The video directory to be detected.')
    parser.add_argument('--noise-type', type=str, default='video_impulse_value',
                        help='Select noise type. [audio_bg_park,audio_color_w,video_gblur,video_impulse_value]')
    parser.add_argument('--save-dir', type=str, default='/home/sharing/disk2/zhangbaozheng/dataset/simsv2/NOISE_IMPULSE',
                        help='The video directory to be detected.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    VIDEO_PATH = args.video_dir
    SAVE_PATH = args.save_dir
    if not os.path.exists('assets/temp'):
        os.makedirs('assets/temp')
    if args.noise_type == 'audio_bg_park':
        audio_bg_park_noise()
    elif args.noise_type == 'audio_color_w':
        audio_color_w_nosie()
    elif args.noise_type == 'video_gblur':
        video_gblur_noise()
    elif args.noise_type == 'video_impulse_value':
        video_impulse_value_noise()