import numpy as np
import pandas as pd
import os
from glob import glob
from msmtu.utils.utils import _read_json, _write_json
import re

# Class to number
class_idx = 0
class_to_idx = {}

mean_sentinel_3 = np.array([1386.6481734178692, 1553.5822105909372, 1650.905763201626, 2807.12981762757,
                            1917.9905670703306, 1408.69896038870349])

std_sentinel_3 = np.array([2274.689559788851, 2130.0496276880863, 2192.7904857619733, 1856.2354811887744,
                           1226.8184704565697, 1090.3276470393764])

mean_modis = np.array([1538.3463272414092, 2695.176373935756, 1255.3621852873657, 1505.6371419346717,
                       1819.9220400236445, 1147.6490066574759])

std_modis = np.array([2030.4308640881818, 1599.3091904898668, 2162.938777320463, 2079.024027380068,
                      1046.2743637031904, 948.6324303065222])


def get_pixel_metadata(df_row, bands_availability, num_metadata: int):
    pixel_metadata = {
        name_meta: int(df_row[name_meta]) if isinstance(df_row[name_meta], np.int64) else df_row[name_meta]
        for name_meta in df_row.index[:num_metadata]
    }

    pixel_metadata['DataAvailability'] = bands_availability

    return pixel_metadata


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

    return l


def get_band_name(csv_path):
    csv_file = csv_path.split(sep='/')[-1]
    csv_name = csv_file.split(sep='.')[0]
    band_name = csv_name.split(sep='_')[-1]

    return band_name


def get_slices(ts_sample_df, window_size):
    seq_length = len(ts_sample_df)
    num_windows = seq_length // window_size  # Integer part
    num_timestamps = window_size * num_windows  # Number of time stamps to take
    leftover = seq_length - num_timestamps

    starting_position = leftover
    ts_subsamples_list = []
    for i in range(num_windows):
        window_starting_position = starting_position + i * window_size
        window_final_position = window_starting_position + window_size

        ts_subsample_df = ts_sample_df.iloc[window_starting_position:window_final_position]

        assert len(
            ts_subsample_df) == window_size, f'Lenght subsample ({len(ts_subsample_df)}) != window size ({window_size})'
        ts_subsamples_list.append(ts_subsample_df)

    return ts_subsamples_list


def parse_delta(masks, dir_, num_dim: int, seq_len: int):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for month in range(seq_len):
        if month == 0:
            deltas.append(np.ones(num_dim))
        else:
            deltas.append(np.ones(num_dim) + (1 - masks[month]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def parse_sample(evals, pixel_metadata, num_sample, json_dir, class_name, test_imp=True):
    evals = (np.array(evals) - mean_sentinel_3) / std_sentinel_3

    shp = evals.shape
    evals = evals.reshape(-1)
    evals = evals.astype(float)

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10)

    values = evals.copy()
    # Indices could be empty if evals=[np.nan, ..., np.nan]
    if len(indices) and test_imp:
        values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))  # Contain random imputation indices to perform evaluation

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {'label': class_name}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    file_name = f'{str(num_sample).zfill(6)}.json'
    class_dir_json = os.path.join(json_dir, class_name)
    file_path = os.path.join(class_dir_json, file_name)

    assert not os.path.exists(file_path), f"The file {file_path} already exists"

    final_data = {
        'metadata': pixel_metadata,
        'ts_data': rec
    }

    _write_json(file_path, final_data, 4)




def parse_csvs_by_month(csv_files, json_dir, class_name, test_imp=False):
    """ Function to parse each csv file.

    Each csv contain the values of all bands for a certain month.
    """

    df_list = []
    band_names = []
    id_pixels_prev = None
    print('CSV FILES')
    print(csv_files)
    for csv in csv_files:
        if 'lat_lon' in csv:  # Skip the files to draw to points in QGIS
            continue

        print(f"Loading {csv} ...")

        # We want to keep blank lines as they represent missing data that we want to impute
        df = pd.read_csv(csv, skip_blank_lines=False).sort_values('pix_id')[
            ['pix_id', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12']]
        id_pixels = df['pix_id'].to_numpy()

        if not (id_pixels_prev is None):
            assert np.array_equal(id_pixels_prev, id_pixels), f'Id pixels are not equal'

        id_pixels_prev = id_pixels

        df_list.append(df)

    band_names = df.columns[1:]  # Get band names
    num_samples = df_list[0].shape[0]

    print(f"Creating json files {class_name}...")
    # Go through each temporal series (ts) sample
    for row_idx in range(num_samples):
        ts_sample_df = pd.DataFrame(columns=band_names, index=list(range(12)))

        for month_idx in range(12):
            df = df_list[month_idx]
            df_row = df.iloc[row_idx, :]
            ts_sample_df.iloc[month_idx, :] = df_row[1:]  # Put df row as a column of the ts sample

        # Parse sample and write to json
        parse_sample(ts_sample_df.to_numpy(na_value=np.nan), None, row_idx, json_dir, class_name, test_imp=test_imp)


def parse_csvs(csv_files, json_dir, class_name, seq_len: int, num_metadata: int):
    """ Function to parse each csv file.

    Each csv contain the values of a band.
    """
    df_list = []
    band_names = []
    id_pixels_prev = None
    for csv in csv_files:
        if 'lat_lon' in csv:  # Skip the files to draw to points in QGIS
            continue

        print(f"Loading {csv} ...")

        # We want to keep blank lines as they represent missing data that we want to impute
        df = pd.read_csv(csv, skip_blank_lines=False).sort_values('IdOfThePixel')
        id_pixels = df['IdOfThePixel'].to_numpy()

        if not (id_pixels_prev is None):
            assert np.array_equal(id_pixels_prev, id_pixels), f'Id pixels are not equal'

        id_pixels_prev = id_pixels

        if len(df.columns) == (seq_len + num_metadata) + 1:
            print(f"Deleting {df.columns[-1]} ...")
            df = df.iloc[:, :-1]

        # Check length of our sequences
        assert len(
            df.columns) == seq_len + num_metadata, f'DF columns ({len(df.columns)}) do not match seq_len + num_metadata ({seq_len + num_metadata})'

        band_names.append(get_band_name(csv))  # Get band names
        df_list.append(df)

    num_samples = df_list[0].shape[0]
    ts_sample_index = df_list[0].columns[num_metadata:]  # Date of each time stamp

    print(f"Creating json files {class_name}...")
    # Go through each temporal series (ts) sample
    for row_idx in range(num_samples):
        ts_sample_df = pd.DataFrame(columns=band_names, index=ts_sample_index)
        bands_availability = []
        for band_idx, band_name in enumerate(band_names):
            df = df_list[band_idx]
            df_row = df.iloc[row_idx, :]
            ts_sample_df[band_name] = df_row[num_metadata:]  # Put df row as a column of the ts sample

            bands_availability.append(float(df_row['DataAvailability']))
        pixel_metadata = get_pixel_metadata(df_row, bands_availability)

        if not (pixel_metadata['NameOfTheClass'] in class_name):
            print(f"Pixel class ({pixel_metadata['NameOfTheClass']}) not equal to class_name ({class_name})")

        # Parse sample and write to json
        parse_sample(ts_sample_df.to_numpy(na_value=np.nan), pixel_metadata, row_idx, json_dir, class_name)


def parse_jsons(class_json, json_dir, class_name, test_imp=True, window_size=262):
    json_data = _read_json(class_json)

    pixels = json_data['Pixels']

    band_names = pixels[0]['Pixel_Metadata']['Temporal_Availability_Percentage'].keys()

    pixel_idx = 0
    for pixel in pixels:

        data_availability = pixel['Pixel_Metadata']['Temporal_Availability_Percentage']
        ts_sample_df = pd.DataFrame(columns=band_names)

        for band_name in band_names:
            pixel_band = np.array(pixel['Pixel_TS'][band_name], dtype=float)
            ts_sample_df[band_name] = pixel_band

            computed_availibility = np.sum(~np.isnan(pixel_band)) * 100 / pixel_band.shape[0]

            assert abs(data_availability[
                           band_name] - computed_availibility) < 0.01, f'Band: {band_name}; DA: {data_availability[band_name]:.4f} != CA: {computed_availibility:.4f}'

        pixel_metadata = pixel['Pixel_Metadata']

        ts_subsamples_list = get_slices(ts_sample_df, window_size)

        for window_idx, ts_subsample_df in enumerate(ts_subsamples_list):
            pixel_metadata['Window_Index'] = window_idx  # Add metadata about window index
            parse_sample(ts_subsample_df.to_numpy(na_value=np.nan), pixel_metadata, pixel_idx, json_dir, class_name,
                         test_imp)
            pixel_idx += 1


def parse_csv_format(dataset_folder, json_dir, test_imp=False):
    classes_dir = sort_nicely(glob(dataset_folder + '*/'))  # Get only directories

    print(classes_dir)

    for class_dir in classes_dir:
        class_name = class_dir.split(sep='/')[-2]
        csv_files = sort_nicely(glob(class_dir + '*.csv'))  # Get only .csvs

        # Create directory where allocate json of each class
        class_dir_json = os.path.join(json_dir, class_name)
        if not os.path.exists(class_dir_json):
            os.makedirs(class_dir_json)

        print(f"Processing {class_dir} directory...")

        parse_csvs_by_month(csv_files, json_dir, class_name, test_imp)


def parse_json_format(dataset_folder, json_dir, test_imp=True, window_size=262):
    classes_json = sorted(glob(dataset_folder + '*.json'))  # Get only directories
    print(classes_json)

    for class_json in classes_json:
        class_name = class_json.split(sep='/')[-1].split(sep='.')[0][1:]

        # Create directory where allocate json of each class
        class_dir_json = os.path.join(json_dir, class_name)
        if not os.path.exists(class_dir_json):
            os.makedirs(class_dir_json)

        print(f"Processing {class_json} json...")

        print(class_name)

        parse_jsons(class_json, json_dir, class_name, test_imp=test_imp, window_size=window_size)


if __name__ == '__main__':
    dataset_folder = './TimeSpec4LULC_Balanced_data/'
    json_dir = './json/LULC_deploy_60'

    dataset_folder = './datasets/Sentinel_Final_Data_500_Threshold_3_Partitions/Test/'
    json_dir = './json/sentinel_aggregated_3_deploy_test'

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # parse_csv_format(dataset_folder_old, json_dir_old)

    # parse_json_format(dataset_folder, json_dir, test_imp=True, window_size=SEQ_LEN)

    parse_csv_format(dataset_folder, json_dir, test_imp=False)
