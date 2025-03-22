import datetime
import pathlib
import pprint

import yaml
import itertools

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

# arguments
notebook_file = pathlib.Path('./Generative Weak Segmentation.ipynb')
hparams_file = pathlib.Path('./hparams_seg.yaml')

save_dir = pathlib.Path('./executed_notebooks')
save_dir.mkdir(parents=True, exist_ok=True)

timeout = 24*3600
kernel = 'python3'

# combinations of hyperparameters
hparams_c = {
    'batch_size': [4],
    'mask_reg': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

hparams_c = [
    dict(zip(hparams_c.keys(), values))
    for values in itertools.product(*hparams_c.values())
]
print(hparams_c)

with open(notebook_file) as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)

with open(hparams_file) as f:
    hparams_yaml = f.read()

print(f'\nExecuting notebook "{notebook_file}"...')
t_start = datetime.datetime.now()

try:
    for i in range(len(hparams_c)):
        loop_t_start = datetime.datetime.now()

        hparams = yaml.safe_load(hparams_yaml)
        hparams.update(hparams_c[i])

        with open(hparams_file, mode='w') as f:
            yaml.dump(hparams, f)
        
        print(f'\nRun {i+1}/{len(hparams_c)} with hparams:')
        pprint.pprint(hparams)
        print('')

        notebook_name = notebook_file.stem
        save_path = save_dir / f'(run {i+1}) {notebook_name}.ipynb'

        try:
            ep.preprocess(nb, {'metadata': {'path': './'}})
        except CellExecutionError as err:
            print(f'\tError occurred during execution: {err}.')
            print(f'\tSee notebook file "{save_path}" for the traceback.')
        finally:
            with open(save_path, mode='w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        
        loop_t_end = datetime.datetime.now()
        print(f'\tDone. Time taken: {loop_t_end - loop_t_start}.')

    t_end = datetime.datetime.now()

    print(f'\n\nAll done. Total time elapsed: {t_end - t_start}.')
    print(f'Average time per run: {(t_end - t_start) / len(hparams_c)}.')

finally:
    with open(hparams_file, mode='w') as f:
        f.write(hparams_yaml)
