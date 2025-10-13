import torch
from typing import Tuple, Callable
from omni_speech.model.builder import load_pretrained_model
import sys
import os
from omni_speech.datasets.preprocess import preprocess_llama_3_v1, preprocess_qwen_2_5_v1
import whisper
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from functools import partial
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
import logging
import librosa
import torchaudio
import json
import onnxruntime
import torchaudio.compliance.kaldi as kaldi
import re
import time
import argparse
import uuid
import glob
import pdb
import subprocess

COSYVOICE_MODEL=""     ## CosyVoice2-0.5B       i.e. /workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B-VocalNet
VOCALNET_MODEL = ""    ##  /root/speechllm_checkpoints/VocalNet-qwen25-7B  VocalNet speech LLM   i.e. ./checkpoints/VocalNet-1B
PROMPT_SPEECH="./omni_speech/infer/cn_prompt.wav"   
# PROMPT_SPEECH="./omni_speech/infer/common_voice_en_2586258.wav"   

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
class SpeechTokenizer:
    def __init__(self, speech_tokenizer_model: str, feat_extractor: Callable, get_tokenizer: Callable, 
                 campplus_model: str, allowed_special: str = 'all',device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = get_tokenizer()
        self.allowed_special = allowed_special
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model, sess_options=option, providers=providers
        )
        
        self.feat_extractor = feat_extractor
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
            self.en_tn_model = EnNormalizer()
    
    def extract_speech_token(self, speech: torch.Tensor, max_duration_sec: int = 30):
        """
        Args:
            speech (torch.Tensor)
            max_duration_sec (int, optional)

        Returns:
            speech_token (torch.Tensor)
            speech_token_len (torch.Tensor)
        """
        sample_rate = 16000  
        max_samples = max_duration_sec * sample_rate

        total_samples = speech.shape[1]
        num_chunks = (total_samples + max_samples - 1) // max_samples  

        speech_tokens = []
        speech_token_lengths = []

        for i in range(num_chunks):
            start = i * max_samples
            end = min(start + max_samples, total_samples)
            chunk = speech[:, start:end] 
            feat = whisper.log_mel_spectrogram(chunk, n_mels=128)  

            input_name_0 = self.speech_tokenizer_session.get_inputs()[0].name
            input_name_1 = self.speech_tokenizer_session.get_inputs()[1].name
            inputs = {
                input_name_0: feat.detach().cpu().numpy(),
                input_name_1: np.array([feat.shape[2]], dtype=np.int32)
            }

            speech_token_output = self.speech_tokenizer_session.run(None, inputs)[0]
            chunk_speech_token = torch.tensor(speech_token_output.flatten().tolist(), dtype=torch.int32).unsqueeze(0).to(self.device)
            chunk_speech_token_len = torch.tensor([chunk_speech_token.shape[1]], dtype=torch.int32).to(self.device)
            speech_tokens.append(chunk_speech_token)
            speech_token_lengths.append(chunk_speech_token_len)

        speech_token = torch.cat(speech_tokens, dim=1)  
        speech_token_len = torch.cat(speech_token_lengths, dim=0).sum().unsqueeze(0)  

        return speech_token, speech_token_len

    def save_tokens(self, speech_token: torch.Tensor, speech_token_len: torch.Tensor, save_path: str):
        np.save(save_path, {'speech_token': speech_token.cpu().numpy(), 'speech_token_len': speech_token_len.cpu().numpy()})
        print(f"Tokens saved to {save_path}")

    def load_tokens(self, load_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(load_path, allow_pickle=True).item() 
        speech_token = torch.tensor(data['speech_token'])
        speech_token_len = torch.tensor(data['speech_token_len'])

        print(f"Tokens loaded from {load_path}")
        return speech_token, speech_token_len
    
    def extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding
    
    def extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len
    def text_normalize(self, text, split=True):
        text = text.strip()
        for token in self.tokenizer.special_tokens['additional_special_tokens']:
            if token in text:
                return text if split is False else [text]
        if contains_chinese(text):
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            else:
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        else:
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        if split is False:
            return text
        return texts

class CosyvoiceVocoder:
    def __init__(self, frontend: SpeechTokenizer, model: CosyVoice2Model, sample_rate: int = 24000):
        frontend = frontend
        self.model = model
        self.sample_rate = sample_rate

    def inference_zero_shot(self, speech_token, prompt_token: torch.Tensor, prompt_feat: torch.Tensor, embedding: torch.Tensor, stream=False, speed=1.0, uuid=None, is_last_speech_chunk=False):
        speech_token = speech_token.squeeze(1) if speech_token.dim() == 3 else speech_token
        prompt_token = prompt_token.squeeze(1) if prompt_token.dim() == 3 else prompt_token

        model_input = {
            'speech_tokens': speech_token,
            'flow_embedding': embedding,
            'prompt_token': prompt_token,
            'prompt_feat': prompt_feat,
            'stream': stream,
            'speed': speed,
            'uuid': uuid,
            'is_last_speech_chunk': is_last_speech_chunk
        }

        for model_output in self.model.tts_direct_update(**model_input):
            yield model_output

class VocalNetModelStream:
    def __init__(self, model_name_or_path: str, vocoder_path: str = COSYVOICE_MODEL, s2s: bool = True, **kwargs):
        self.s2s = s2s
        self.model_name_or_path = model_name_or_path
        self.empty = True
        self.vocoder_path = vocoder_path

        self.temperature = kwargs.get('temperature', 0.1)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.1)
        self.top_p = kwargs.get('top_k', 0.0)
        self.streaming = kwargs.get('streaming', False)

        self.audio_dir = None
        self.empty = True

        self.txt_token_num = kwargs.get('txt_token_num', 5)
        self.speech_token_num = kwargs.get('speech_token_num', 15)
        self.reset_interval = kwargs.get('reset_interval', 50)

    def __initilize__(self):
        if self.empty:
            self.empty = False
            self.tokenizer, self.model, _ = load_pretrained_model(self.model_name_or_path, s2s=self.s2s)
            self.model.tokenizer = self.tokenizer
            self.__init_vocoder__()


    def __init_vocoder__(self):
        model_dir = self.vocoder_path
        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        frontend = SpeechTokenizer(
            speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v2.onnx",
            feat_extractor=configs['feat_extractor'],
            get_tokenizer=configs["get_tokenizer"],
            allowed_special=configs['allowed_special'],
            campplus_model=f"{model_dir}/campplus.onnx"
        )

        model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'])

        model.load(
            f'{model_dir}/llm.pt',
            f'{model_dir}/flow.pt',
            f'{model_dir}/hift.pt'
        )

        self.cosy_vocoder = CosyvoiceVocoder(frontend=frontend, model=model)

        prompt_wav = PROMPT_SPEECH
        prompt_sr = 16000 

        prompt_speech_16k = self.postprocess(load_wav(prompt_wav, prompt_sr))

        prompt_token, prompt_token_len = frontend.extract_speech_token(prompt_speech_16k)
        resample_rate = 24000  
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=prompt_sr, new_freq=resample_rate)(prompt_speech_16k)

        speech_feat, speech_feat_len = frontend.extract_speech_feat(prompt_speech_resample)
        if resample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), prompt_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            prompt_token, prompt_token_len[:] = prompt_token[:, :token_len], token_len
        embedding = frontend.extract_spk_embedding(prompt_speech_16k)

        self.prompt_token = prompt_token
        self.speech_feat = speech_feat
        self.embedding = embedding
        

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440, max_val = 0.8):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(24000 * 0.2))], dim=1)
        return speech

    def set_audio_dir(self, output_dir):
        self.audio_dir = output_dir

    def __call__(self, messages: list):
        wav_file = messages[0]['path']

        global_uuid = str(uuid.uuid1())
        speech = whisper.load_audio(wav_file)

        raw_len = len(speech)
        speech = whisper.pad_or_trim(speech)
        padding_len = len(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).unsqueeze(0)
        speech_length = round(raw_len / padding_len * 3000 + 0.5)
        speech_length = torch.LongTensor([speech_length])

        conversation = [{"from": "human", "value": "<speech>", "path": f"{wav_file}"}]
        if 'qwen' in self.model_name_or_path.lower():
            input_ids = preprocess_qwen_2_5_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([198, 151644, 77091, 198], device=input_ids.device)]).unsqueeze(0)
        else:
            input_ids = preprocess_llama_3_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([128006, 78191, 128007, 271], device=input_ids.device)]).unsqueeze(0)


        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        step = 0
        full_generated_text_idx = torch.empty([1,0], device='cuda', dtype=torch.int64)
        self.model.reset_streaming_state()
   
        sample_rate = None
        speech_list = []               

        with torch.inference_mode():
            if self.s2s:
                for generated_text_idx, units_pred, is_last_speech_chunk in self.model.streaming_generate_mtp(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    temperature=0.1,
                    top_p=0.1,
                    top_k=0,
                    use_cache=True,
                    infer_mtp_token_num=2,
                    txt_token_num=self.txt_token_num,
                    speech_token_num = self.speech_token_num,
                    reset_interval=self.reset_interval
                ):  
                    if generated_text_idx is not None:
                        full_generated_text_idx = torch.cat([full_generated_text_idx, generated_text_idx], dim=-1)

                    for unit_tensor in units_pred:
                        sub_step = 0
                        output_units = unit_tensor
                        
                        for output in self.cosy_vocoder.inference_zero_shot(
                            speech_token=output_units,
                            prompt_token=self.prompt_token,
                            prompt_feat=self.speech_feat,
                            embedding=self.embedding,
                            stream=True,
                            speed=1,
                            uuid=global_uuid,
                            is_last_speech_chunk=is_last_speech_chunk
                        ):
                            speech = output['tts_speech']

                            if sample_rate is None:
                                sample_rate = self.cosy_vocoder.sample_rate

                            speech_list.append(speech)
                            sub_step += 1                        
                        step += 1

        merged_audio_path = None
        if speech_list:
            base_name = os.path.basename(wav_file).replace('mp3', 'wav')
            merged_audio_path = os.path.join(self.audio_dir, base_name)
            final_speech = torch.cat(speech_list, dim=-1)
            torchaudio.save(merged_audio_path, final_speech.cpu(), sample_rate)
        
        self.cosy_vocoder.model.reset_state(global_uuid)
        full_generated_text = self.tokenizer.decode(full_generated_text_idx[0].cpu().numpy(), skip_special_tokens=True)
        return {"text": full_generated_text.strip(), 'audio': merged_audio_path} 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VocalNet infer')
    parser.add_argument('--query_audio', type = str, default="./omni_speech/infer/alpaca_cn_query.wav")
    parser.add_argument('--s2s', action='store_true', default=True)
    parser.add_argument('--save_dir', default="./generated_audio")
    args = parser.parse_args()

    audio_messages = [{"role": "user", "content": "<speech>","path": args.query_audio}]
    print("Initialized vocalnet")
    vocalnet = VocalNetModelStream(VOCALNET_MODEL, s2s=args.s2s)
    vocalnet.__initilize__()
    vocalnet.set_audio_dir(args.save_dir)

    response = vocalnet.__call__(audio_messages)
    print(response)


