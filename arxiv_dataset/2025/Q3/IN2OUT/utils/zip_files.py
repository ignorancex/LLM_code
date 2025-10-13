import os
import zipfile
from tqdm import tqdm

data_dir = '../datas/youtube-vos/JPEGOriginal'

def zip_files_in_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

for vid in tqdm(os.listdir(data_dir)):
    folder = os.path.join(data_dir, vid)
    zip_files_in_folder(folder, f'../datas/youtube-vos/JPEGImages/{vid}.zip')
    