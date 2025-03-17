import sys
import os
if __name__=="__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    sys.path.pop(0)

import tensorflow_datasets as tfds
import datasets

if __name__=="__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    ds_name = sys.argv[1]
    split = sys.argv[2]
    if not ds_name.split('/')[0] in tfds.list_builders():
        print('Registered datasets:')
        for d in tfds.list_builders():
            print(d)
        sys.exit(1)
    ds, info = tfds.load(ds_name, split=split, with_info=True)
    fig = tfds.show_examples(ds, info)
    fig.savefig('out.png')