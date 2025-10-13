import json
import argparse
import numpy as np
from pathlib import Path


def saved_embedding_to_chungus(embeddings_file, settings_file, create_blank=False):
    """ Save embeddings file to a CHUNGUS-ready file (for use in ROS CHUNGUS package)
    
    :param embeddings_file: embeddings file
    :param settings_file: settings file
    :param create_blank: if true, will create blank embeddings file
    :returns: dictionary with embeddings data
    """

    with open(settings_file, 'r') as f:
        embedding_settings = json.load(f)

    if create_blank:
        embeddings = np.zeros((0,2,384), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
        cls_tokens = np.zeros((0,384), dtype=np.float32)
        images = []
        timings = []
        is_initial_label = []
        
        novelty_scores = []
        is_novel = []
        is_initial_image = []
        training_times = []
    else:
        raw = np.load(embeddings_file, allow_pickle=True).item()['data']
        assert(len(raw) == 1)
        raw = raw[0]
        embeddings = raw['embeddings'].reshape(-1, *raw['embeddings'].shape[2:])
        labels = raw['labels'].reshape(-1, *raw['labels'].shape[2:])
        cls_tokens = raw['cls_tokens'].reshape(-1, *raw['cls_tokens'].shape[2:])
        images = sorted([Path(p[0]).name for p in raw['image_files']] + [Path(p[1]).name for p in raw['image_files']])
        timings = [float('nan')] * len(labels)
        is_initial_label = [True] * len(labels)
        
        novelty_scores = [float('nan')] * len(images)
        is_novel = [False] * len(images)
        is_initial_image = [True] * len(images)
        training_times = []
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'cls_tokens': cls_tokens,
        'image_files': images,
        'timings': timings,
        'is_initial_label': is_initial_label,
        'novelty_scores': novelty_scores,
        'is_novel': is_novel,
        'is_initial_image': is_initial_image,
        'training_times': training_times,
        'settings': {
            'res_x': embedding_settings['res_x'],
            'res_y': embedding_settings['res_y'], 
            'model': embedding_settings['model'], 
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--embedding_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default='./chungus_embeddings')
    parser.add_argument("--create_blank", action='store_true')
    args = parser.parse_args()

    # Extract settings and save embeddings
    settings = vars(args)
    result = saved_embedding_to_chungus(
        Path(settings['embedding_folder']) / Path('embeddings.npy'),
        Path(settings['embedding_folder']) / Path('settings.json'),
        create_blank=args.create_blank
    )
    
    # Show some information about the data
    print("Generated data...")
    for k in result.keys():
        print("{} -> {} (shape = {})".format(
            k, len(result[k]), result[k].shape if type(result[k]) == np.ndarray else None
        ))
        if type(result[k]) == dict:
            for k2 in result[k].keys():
                print("\t{} -> {}".format(k2, result[k][k2]))
    
    # Save output file
    output_file = Path(settings['output_folder']) / Path('{}{}.npy'.format(
        Path(settings['embedding_folder']).name,
        '_BLANK' if args.create_blank else ''
    ))
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    np.save(str(output_file), result)
    print("Saved to {}".format(output_file))
