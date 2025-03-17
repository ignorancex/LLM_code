import subprocess
import os
import sys
from copy import deepcopy
from pathlib import Path
import tempfile

import torch

from translate import get_lm_args_parser, translate as translate_args

def write_nlines(fread, fwrite, num_lines):
    lines = []
    num = 0
    for line in fread:
        fwrite.write(line)
        num += 1
        if num >= num_lines:
            break
    return lines

def del_flag(new_cmd, flag):
    idx_n_gpus = new_cmd.index(flag)
    del new_cmd[idx_n_gpus+1]
    del new_cmd[idx_n_gpus]


def get_new_cmd(orig_cmd, part, out_name):
    new_cmd = orig_cmd.copy()

    new_cmd[0] = 'src/zsmt/translate.py'

    del_flag(new_cmd, '--out_parts')
    del_flag(new_cmd, '--n_gpus')

    idx_input = new_cmd.index('--input')
    new_cmd[idx_input+1] = str(part)

    idx_output = new_cmd.index('--output')
    new_cmd[idx_output+1] = str(out_name)

    try:
        idx_ms = new_cmd.index('--max-sents')
        new_cmd[idx_ms+1] = str(-1)
    except ValueError:
        pass

    return ['python'] + new_cmd

if __name__ == "__main__":
    f_tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir = Path(f_tmp_dir.name)

    parser = get_lm_args_parser()
    parser.add_argument('--out_parts', type=Path, required=True)
    parser.add_argument('--n_gpus', type=int, default=-1)
    args = parser.parse_args()

    args.out_parts.mkdir(exist_ok=True, parents=True)

    dev_count = torch.cuda.device_count()
    n_gpus = dev_count if args.n_gpus == -1 else args.n_gpus
    n_gpus = min(dev_count, n_gpus)

    stem = f'{args.input_path.stem}.part'
    if args.max_sents != -1:
        f_temp = tempfile.NamedTemporaryFile('w+')
        # f_temp=open('first.txt', 'w+')
        with args.input_path.open('r') as fread:
            write_nlines(fread, f_temp, args.max_sents)
        f_temp.flush()
        input_path = f_temp.name
    else:
        input_path = args.input_path

    prefix = tmp_dir / stem
    subprocess.call(f'split {input_path} {prefix} -n l/{n_gpus}'.split(' '))

    orig_cmd = sys.argv

    procs = []
    parts = sorted(list(tmp_dir.glob(f'{stem}??')))
    out_names = [args.out_parts / f'{part.name}.pred.txt' for part in parts]
    out_names = [str(x) for x in out_names]
    cmds = [get_new_cmd(orig_cmd, part, out_name) for part, out_name in zip(parts, out_names)]
    for gpu_id, cmd in zip(range(n_gpus), cmds):
        print(' '.join(cmd))
        proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id)))
        procs.append(proc)

    [proc.wait() for proc in procs]

    with args.output_path.open('w') as f:
        subprocess.call(f'cat {" ".join(out_names)}'.split(' '), stdout=f)

    print(f'saved to {args.output_path}')
    print(f'consider deleting the parts at {args.out_parts}')

    if args.max_sents != -1:
        f_temp.close()
