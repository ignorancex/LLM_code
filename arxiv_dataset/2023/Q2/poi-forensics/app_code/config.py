import os
import yaml

DEFAULT_OPT = {
    'resources_path': None,
    'fps': 25,
    'read_stride': 96,
    'rec_stride': 32,
    'det_stride': 12,
    'face_det': {
        'size_threshold': 75,
        'score_threshold': 0.7,
        'iou_threshold': 0.4,
    },
    'audio': {
        'sampling_rate': 16000,
        'norm_target_dBFS': -30,
        'num_fft': 512,
        'window_step': 10,
        'window_length': 25,
    },
    'model': 'poiforensics',
    'percentile': 5,
    'final_mean': 7,
    'dist_normalization': None,
    'output_ffmpeg_params': {
        '-c:v': 'libx264', '-profile:v': 'high', '-level:v': '4.0',
        '-pix_fmt': 'yuv420p', '-crf': '35',
    },
}


def create_opt(**opt):
    opt = {key: opt.get(key, DEFAULT_OPT[key]) for key in DEFAULT_OPT}
    if not isinstance(opt['model'], dict):
        file_model = './config/%s.yaml' % opt['model']
        with open(file_model) as fid:
            opt['model'] = yaml.load(fid, Loader=yaml.FullLoader)['model']
    if opt['dist_normalization'] is None:
        opt['dist_normalization'] = opt['model']['type']=='poi_forensics'
    return opt


def get_extraction_opt(opt):
    return {key: opt[key] for key in [
        'fps',
        'audio',
        'model',
    ]}

