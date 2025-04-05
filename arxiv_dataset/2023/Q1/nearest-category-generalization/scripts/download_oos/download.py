import os
import shutil

import requests
from PIL import Image
from tqdm import tqdm
from mkdir_p import mkdir_p

#target_dir = "./data/oos_examples/groot/"
#filename = "./scripts/download_oos/groot_urls.txt"
#target_dir = "./data/oos_examples/chewbacca/"
#filename = "./scripts/download_oos/chewbacca_urls.txt"
target_dir = "./data/oos_examples/jarjar/"
filename = "./scripts/download_oos/jarjar_urls.txt"
#target_dir = "./data/oos_examples/c3po/"
#filename = "./scripts/download_oos/c3po_urls.txt"

mkdir_p(target_dir)
with open(filename, "r") as f:
    urls = f.readlines()

for i, url in enumerate(urls):
    print([url])
    local_image_filename = "temp"

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(url[:-1], stream = True)
    if r.status_code == 200:
        r.raw.decode_content = True
        # Open a local file with wb ( write binary ) permission.
        with open(local_image_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded: ',local_image_filename)
    else:
        print('Image Couldn\'t be retreived')
        print(r.status_code)

    im = Image.open(local_image_filename)
    im.save(os.path.join(target_dir, f"{i:06d}.png"), "png")
    os.remove(local_image_filename)

