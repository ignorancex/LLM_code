import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main(opt):
    if not opt.root_dir:
        raise Exception('root_dir for generation not specified')
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1. NOTE : probably not required
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    
    scenario_ref = f'{opt.source[0]}2{opt.target[0]}'
    opt.root_dir = os.path.join(opt.root_dir, opt.dataset, scenario_ref)
    if not os.path.exists(opt.root_dir):
        os.makedirs(opt.root_dir, exist_ok=True)
    
    for i, data in tqdm(enumerate(dataset), total=len(dataset)): # tqdm len works because batch size is 1
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        im = Image.fromarray(util.tensor2im(visuals['fake_B']))
        rel_path = '/'.join(img_path[0].split('/')[-2:])
        save_path = Path(opt.root_dir) / rel_path
        os.makedirs(save_path.parent, exist_ok=True)
        im.save(save_path)
        
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(opt)