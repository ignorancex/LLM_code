# Generate pitch and stretch random changes every 100 ms
# across a word or phrase
import sys
import tomli
import json

sys.path.insert(1, '/home/rosie/Documents/vocal_ambiguity/interspeech/stimulus_generation/cleese_clone')
import cleese_stim as cleese
from cleese_stim.engines import PhaseVocoder


BASE_SOUND = "./sounds/tts/phrase_peel_pill_final.wav"
LANGUAGE = 'english'
EXPERIMENT = 'experiment_phrase'
CONFIG_DICT = {
  'stretch': "./configs/random_stretch_profile.toml",
  'pitch': "./configs/random_pitch_profile.toml",
}
NUM_FILES = 1000

def generateCustomBP(transform, duration, config, config_file):
  eqFreqVec = None

  # bpf timepoints for stretch and pitch
  bpf_time, num_points, end_on_trans = PhaseVocoder.create_BPF_time_vec(
      duration,
      config[transform]
  )

  # create a bpf for each of the transformations
  bpf_list = (PhaseVocoder.create_BPF(
      transform,
      config_file,
      bpf_time,
      num_points,
      end_on_trans,
      eqFreqVec
  ))

  return bpf_list

def generate_file(input_file, config_dict, output_file, audio_path, bpf_path):

  wave_in, sr, _ = PhaseVocoder.wavRead(input_file)
  duration = len(wave_in) / float(sr)

  f = open(config_dict['stretch'], "rb")
  stretch_config = tomli.load(f)
  f = open(config_dict['pitch'], "rb")
  pitch_config = tomli.load(f)

  bpf_dict = {}

  bpf_list = generateCustomBP('pitch', duration, pitch_config, config_dict['pitch'])
  wave_out,bpf_out = cleese.process_data(
      PhaseVocoder,
      wave_in,
      config_dict['pitch'],
      sample_rate=sr,
      BPF=bpf_list
  )
  wave_in = wave_out
  bpf_dict['pitch'] = str(bpf_out)

  bpf_list = generateCustomBP('stretch', duration, stretch_config, config_dict['stretch'])
  wave_out,bpf_out = cleese.process_data(
      PhaseVocoder,
      wave_in,
      config_dict['stretch'],
      sample_rate=sr,
      BPF=bpf_list
  )
  bpf_dict['stretch'] = str(bpf_out)

  PhaseVocoder.wavWrite(wave_out, audio_path+output_file+'.wav', sr)
  with open(bpf_path + output_file+'.json', 'w') as f:
    json.dump(bpf_dict, f)

audio_path = f'../{EXPERIMENT}/sounds/{LANGUAGE}/audio/'
bpf_path = f'../{EXPERIMENT}/sounds/{LANGUAGE}/bpf/'

for i in range(NUM_FILES):
  output_file  = f'{LANGUAGE}_{i+1}'
  generate_file(BASE_SOUND, CONFIG_DICT, output_file, audio_path, bpf_path)