import os

def _get_filenames_with_labels(mode, data_dir, split_dir):
  """
  Returns all training or test files in the data directory with their
  respective labels.
  """
  if mode == 'train':
    scenario_list_file = os.path.join(split_dir, 'train.txt')
  elif mode == 'eval':
    scenario_list_file = os.path.join(split_dir, 'eval.txt')
  elif mode == 'test':
    scenario_list_file = os.path.join(split_dir, 'test.txt')
  else:
    raise ValueError("Mode %s is not supported!" % mode)
  with open(scenario_list_file) as f:
    scenario_list = f.read().split('\n')
    scenario_list.pop()
  

  filenames = []
  labels = []
  for i, scenario in enumerate(scenario_list):
    if (i+1) % 100 == 0:
      print("%s / %s : %s" % (i+1, len(scenario_list), scenario))
    scenario_dir = os.path.join(data_dir, 'recordings', scenario)
    # if "vcom=0" in scenario and "vpsf=0" in scenario: # stable scenario
    #   label = 0.0
    # else: # unstable scenario
    #   label = 1.0
    h=-1
    if 'h=2' in scenario:
      h=2
    elif 'h=3' in scenario:
      h=3
    elif 'h=4' in scenario:
      h=4
    elif 'h=5' in scenario:
      h=5
    elif 'h=6' in scenario:
      h=6
    for img_file in filter(
        lambda f: f.startswith('rgb-') and f.endswith('-mono-0.png'),
        os.listdir(scenario_dir)):
      filenames.append(os.path.join(scenario_dir, img_file))
      labels.append(h-2)

  return filenames, labels