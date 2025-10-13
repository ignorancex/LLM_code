import os
import shutil

os.makedirs("/root/.config/kaggle", exist_ok=True)

shutil.copy("kaggle.json", "/root/.config/kaggle/kaggle.json")
