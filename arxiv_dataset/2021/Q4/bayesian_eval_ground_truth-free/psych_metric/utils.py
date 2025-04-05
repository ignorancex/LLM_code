import math
import numpy as np
import random

try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        shell = 'jupyter'
    if 'terminal' in ipy_str:
        shell =  'ipython'
except:
    shell = 'terminal'

if shell == 'jupyter':
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def expand_box(box, scale):
    """expand_box

    :param box: [int, ...]: x,y,w,h
    :param scale: float
    """
    box_scale = math.sqrt(scale)
    x,y,w,h = box

    nh, nw = box_scale * h, box_scale * w
    nx, ny = x - (nw - w) / 2., y - (nh - h) / 2.
    nbox = [int(np.rint(i)) for i in [nx,ny,nw,nh]]
    return nbox

def make_box_square(box, keep_large_side=True):
    x,y,w,h = box
    if (h > w and keep_large_side) or (w > h and not keep_large_side) :
        delta = (h - w) // 2
        x -= delta
        w = h
    elif (h > w and not keep_large_side) or (w > h and keep_large_side) :
        delta = (w - h) // 2
        y -= delta
        h = w
    else:
        # already the same size
        pass

    return x,y,w,h

def cut_out_box(arr, box, pad_mode='edge'):
    ah,aw = arr.shape[:2]
    bx,by,bw,bh = box
    bx1,by1,bx2,by2 = bx,by,bx+bw,by+bh

    pad_x1 = 0 - bx1 if bx1 < 0 else 0
    new_x1 = max(0, bx1)

    pad_y1 = 0 - by1 if by1 < 0 else 0
    new_y1 = max(0, by1)

    pad_x2 = bx2 - aw if bx2 > aw else 0
    new_x2 = min(aw, bx2)

    pad_y2 = by2 - ah if by2 > ah else 0
    new_y2 = min(ah, by2)

    if sum([pad_x1, pad_y1, pad_x2, pad_y2]) == 0:
        return arr[by1:by2, bx1:bx2]
    else:
        arr = arr[new_y1:new_y2, new_x1:new_x2]
        if len(arr.shape) == 3:
            padding = ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0))
        elif len(arr.shape) == 2:
            padding = ((pad_y1, pad_y2), (pad_x1, pad_x2))
        return np.pad(arr, padding, mode=pad_mode)

def jitter_box(box, pw=0.1, ph=0.1):
    x,y,w,h = box
    dw = random.uniform(-pw*w, pw*w)
    dh = random.uniform(-ph*h, ph*h)
    x, y = x + dw, y + dh
    x, y = int(np.rint(x)), int(np.rint(y))
    return (x,y,w,h)
