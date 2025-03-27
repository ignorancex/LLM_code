import sys
import yaml
import torch
import numpy as np
sys.path.append('..')
from PIL import Image
from solvers.solver import get_solvers
from utils.util import sample_noise


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]

if __name__ == "__main__":
    model_dir = "<path of model>"  # the path you save your model
    config = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)
    solver = get_solvers(config)
    solver.G.load_state_dict(torch.load(model_dir, map_location=solver.device))
    solver.G.to(solver.device)
    solver.G.eval()

    z = sample_noise(batch_size=50000, device=solver.device)[0]
    batches = make_batches(50000, 25)
    fake_images = []
    for batch_idx, (batch_start, batch_end) in enumerate(batches):
        noise_batch = z[batch_start:batch_end].to(solver.device)
        out = solver.G(noise_batch).detach().cpu().numpy()
        out = np.multiply(np.add(np.multiply(out, 0.5), 0.5), 255).astype('int32')
        fake_images.append(out)
    fake_images = np.vstack(fake_images)
    fake_images = np.transpose(fake_images, (0, 2, 3, 1))
    im = Image.fromarray(np.uint8(fake_images[0]))
    np.save('./generated_imgs', fake_images)
