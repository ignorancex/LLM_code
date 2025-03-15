import torch
import sys
import os
import argparse
sys.path.append(os.path.dirname(sys.path[0]))

from utils.misc import reload_for_eval
from utils.decode import decode_one_audio
from dataloader.dataloader import DataReader
import yamlargparse
import soundfile as sf
import warnings
from networks import network_wrapper

warnings.filterwarnings("ignore")

def inference(args):
    device = torch.device('cuda') if args.use_cuda==1 else torch.device('cpu')
    print(device)
    print('creating model...')
    model = network_wrapper(args).se_network
    model.to(device)

    print('loading model ...')
    reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
    model.eval()
    with torch.no_grad():

        data_reader = DataReader(args)
        output_wave_dir = args.output_dir
        if not os.path.isdir(output_wave_dir):
            os.makedirs(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            input_audio, wav_id, input_len, scalar = data_reader[idx]
            print(f'audio: {wav_id}')
            output_audio = decode_one_audio(model, device, input_audio, args) 
            output_audio = output_audio[:input_len] * scalar
            sf.write(os.path.join(output_wave_dir, wav_id), output_audio, args.sampling_rate)
    print('Done!')
if __name__ == "__main__":
    # parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser = yamlargparse.ArgumentParser("Settings")
    parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile)

    parser.add_argument('--mode', type=str, default='inference', help='run train or inference')        
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoint_dir/FRCRN', help='the checkpoint dir')
    parser.add_argument('--input-path', dest='input_path', type=str, help='input dir or scp file for saving noisy audio')
    parser.add_argument('--output-dir', dest='output_dir', type=str, help='output dir for saving processed audio')
    parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='the num gpus to use')
    parser.add_argument('--network', type=str, help='select speech enhancement models: FRCRN_SE_16K, MossFormer2_SE_48K')
    parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000)
    parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=int, default=60,
                        help='the max length (second) for one-time pass decoding')
    parser.add_argument('--decode-window', dest='decode_window', type=int, default=1,
                        help='segmental decoding window length (second)')

    ## FFT Parameters
    parser.add_argument('--window-len', dest='win_len', type=int, default=400, help='the window-len in enframe')
    parser.add_argument('--window-inc', dest='win_inc', type=int, default=100, help='the window include in enframe')
    parser.add_argument('--fft-len', dest='fft_len', type=int, default=512, help='the fft length when in extract feature')
    parser.add_argument('--num-mels', dest='num_mels', type=int, default=60, help='the number of mels used for mossformer')
    parser.add_argument('--window-type', dest='win_type', type=str, default='hamming', help='the window type in enframe, include hamming and None')
    args = parser.parse_args()
    print(args)

    inference(args)
