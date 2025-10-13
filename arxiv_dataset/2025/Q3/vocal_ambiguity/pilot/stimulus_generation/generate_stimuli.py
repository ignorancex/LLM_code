import sys
import numpy as np

sys.path.insert(1, 'C:\\Users\\ptut0\Documents\\vocal_ambiguity\\cleese_clone\\')

import cleese_stim as cleese
from cleese_stim.engines import PhaseVocoder
import tomli
import json

import os

#### DEFINE THE FOLLOWIN PARAMS AS PER USE CASE ####
#The timepoints for each of the words of intrest for each phrase
time_points = {
  "sat" : [2.50, 2.60],
  "set" : [2.84, 2.94],
  "beat" : [3.51, 3.62],
  "bit" : [2.83, 2.93],
  "fou" : [0.62, 0.79],
  "fut" : [0.62, 0.84],
  "brillant" : [3.65, 3.81],
  "bruyant" : [3.68, 3.89]
}

parent_dir = "C:\\Users\\ptut0\\Documents\\vocal_ambiguity\\stimulus_generation"

input_file = "./sounds/110_flat_groupe_brillant.wav"

config_dict = {
  'stretch': "./configs/random_stretch_profile.toml",
  'pitch': "./configs/random_pitch_profile.toml",
  'eq': "./configs/random_timbre_profile.toml"
}

transforms = ['eq', 'stretch', 'pitch']

num_files = 600

#######################################################

def generate_file(input_file, transforms, config_dict, output_file, audio_path, bpf_path, time_points=None):
  '''
    apply any of eq, stretch and pitch transformations to a file as per the provided config
    Stretch and pitch will be applied to the entire file as per the config
    eq will be applied only given the provided time points if provided
  '''

  wave_in, sr, _ = PhaseVocoder.wavRead(input_file)
  duration = len(wave_in) / float(sr)
  
  for key in config_dict:
    f = open(config_dict[key], "rb")
    if key == 'stretch':
      stretch_config = tomli.load(f)
    if key == 'pitch':
      pitch_config = tomli.load(f)
    if key == 'eq':
      eq_config = tomli.load(f)
  
  word = input_file.split("_")[-1]
  word = word.split(".")[0]

  if time_points != None:
    time_points_word = np.array(time_points[word]) 
  
  def generateCustomBP(transform, duration, config, config_file, time_points_word=None,):
    '''
    generate the bpf given the desired transformations
    '''
    
    eqFreqVec = None
    if transform == 'eq':
      eqFreqVec = PhaseVocoder.createBPFfreqs(config)

      if time_points_word.any() == None:
        print("No timepoints provided the eq transformation will apply to the entire file ")
      else:
        # timpoints for eq, only the word of interest
        bpf_time = [0, time_points_word[0], time_points_word[0], time_points_word[1], time_points_word[1], duration]

        # create a bpf for each of the transformations
        bpf_list = (PhaseVocoder.create_BPF(
          transform,
          config_file,
          bpf_time,
          6,
          0,
          eqFreqVec
        ))

        bpf_list[0, 3:] = np.zeros(len(bpf_list[0, 3:]))
        bpf_list[1, 3:] = np.zeros(len(bpf_list[1, 3:]))
        bpf_list[4, 3:] = np.zeros(len(bpf_list[4, 3:]))
        bpf_list[5, 3:] = np.zeros(len(bpf_list[5, 3:]))

        return bpf_list
  
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

    
  bpf_dict = {}

  # run the transformations
  if 'eq' in transforms:
    bpf_list = generateCustomBP('eq', duration, eq_config, config_dict['eq'], time_points_word)
    wave_out,bpf_out = cleese.process_data(
        PhaseVocoder,
        wave_in,
        config_dict['eq'],
        sample_rate=sr,
        BPF=bpf_list
    )
    wave_in = wave_out
    bpf_dict['eq'] = str(bpf_out)

  if 'pitch' in transforms:
    bpf_list = generateCustomBP('pitch', duration, pitch_config, config_dict['pitch'], time_points_word)
    wave_out,bpf_out = cleese.process_data(
        PhaseVocoder,
        wave_in,
        config_dict['pitch'],
        sample_rate=sr,
        BPF=bpf_list
    )
    wave_in = wave_out
    bpf_dict['pitch'] = str(bpf_out)

  if 'stretch' in transforms:
    bpf_list = generateCustomBP('stretch', duration, stretch_config, config_dict['stretch'], time_points_word)
    wave_out,bpf_out = cleese.process_data(
        PhaseVocoder,
        wave_in,
        config_dict['stretch'],
        sample_rate=sr,
        BPF=bpf_list
    )
    wave_in = wave_out
    bpf_dict['stretch'] = str(bpf_out)

  # write audio file
  PhaseVocoder.wavWrite(wave_out, audio_path+output_file+'.wav', sr)
  # write the bpf
  with open(bpf_path + output_file+'.json', 'w') as f:
    json.dump(bpf_dict, f)


if __name__ == "__main__":
  base_file = os.path.splitext(os.path.basename(input_file))[0]
  file_path = os.path.join(parent_dir, base_file)
  audio_path = os.path.join(parent_dir, base_file, 'audio\\') 
  bpf_path = os.path.join(parent_dir, base_file, 'bpf\\')
  if not os.path.exists(file_path):
    os.mkdir(file_path)
    os.mkdir(audio_path)
    os.mkdir(bpf_path)
  
  for i in range(num_files):
    output_file  = base_file + f'_{i+1}'
    generate_file(input_file, transforms, config_dict, output_file, audio_path, bpf_path,  time_points)