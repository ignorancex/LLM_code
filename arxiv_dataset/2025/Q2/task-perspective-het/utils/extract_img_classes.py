import os
import numpy as np
import pandas as pd
from glob import glob
import argparse

def process_npy_files(directory, client):

  data = []
  client_dirs = os.listdir(directory)
  client_dirs.sort()
  for file in client_dirs:
    if file.endswith(".npy"):
      file_path = os.path.join(directory, file)
      data_array = np.load(file_path)
      class_idx = np.argmax(data_array)
      data.append([file[:-16]+"rgb.png", class_idx])

  df = pd.DataFrame(data, columns=["filepath", "class_idx"])
  df.to_csv(os.path.join(directory, f"scene_classes.csv"), index=False)

def parse_args():
  argparser = argparse.ArgumentParser("Extracting scene classes from client datasets")
  argparser.add_argument('--client_file_name', type=str, help="Path to the client information .csv file")
  argparser.add_argument('--data_path', type=str, help="Path to the data files")

  args = argparser.parse_args()

  return args

if __name__ == "__main__":
  args = parse_args()
  client_file_path = os.path.join(args.data_path, args.client_file_name)
  clients_df = pd.read_csv(client_file_path, delimiter=",")
  clients = clients_df.id.values
  for client in clients:
    print(client)
    pathname = os.path.join(args.data_path, f'class_scene/taskonomy/{client}/')
    process_npy_files(pathname, client)