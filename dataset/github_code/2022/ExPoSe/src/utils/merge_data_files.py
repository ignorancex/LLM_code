import argparse
import glob
import pickle

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    args = parser.parse_args()

    files = glob.glob(args.files)

    print('Files: ')
    print(files)
    data = {
        'states': [],
        'actions': [],
        'values': []
    }

    for file in tqdm(files):
        with open(file, 'rb') as f:
            file_data = pickle.load(f)

        data['states'].extend(file_data['states'])
        data['actions'].extend(file_data['actions'])
        data['values'].extend(file_data['values'])

    print('Total samples: ', len(data['actions']))
    with open(args.save, 'wb') as f:
        pickle.dump(data, f)
        f.close()
