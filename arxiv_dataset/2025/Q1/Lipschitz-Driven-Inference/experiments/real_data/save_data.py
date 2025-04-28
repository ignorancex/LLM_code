from pathlib import Path
import os
import numpy as np
import pandas as pd
"""
Dataset is downloaded from a dropbox provided by Kerri Lu. 
"Quantifying uncertainty in area and regression coefficient estimation from remote sensing maps"
by Kerri Lu, Stephen Bates and Sherrie Wang.
https://arxiv.org/abs/2407.13659
The ground truth tree cover was labelled manually for their paper,
 and they should be cited if this dataset is used
"""
boundary_map = dict(
    south=dict(
        min_lat = 25, max_lat = 38, min_lon = -100, max_lon = -75, test_frac = 0.5
    ),
    west=dict(
        min_lat = 25, max_lat = 50, min_lon = -125, max_lon = -110, test_frac = 0.5
    )
)  

datadir = Path(Path(__file__).parent, "data")
tree_cover_data = Path(datadir, 'tree_cover_lin_reg_1k_ground_truth_final.csv')


def download_data():
    # Check if the data is already downloaded. if it is not, download it
    os.makedirs(datadir, exist_ok=True)
    if not os.path.exists(tree_cover_data):
        import requests
        url = 'https://www.dropbox.com/scl/fo/akdq400xjyhih3z2oka9k/AAIs9pgdj-iuLlQiSnlIr3Y/tree_cover_lin_reg_1k_ground_truth_final.csv?rlkey=c9uvjxs5hv2mfku0fppsrgdwl&dl=1'
        r = requests.get(url, allow_redirects=True)
        open(tree_cover_data, 'wb').write(r.content)

  
def save_data(region: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, test_frac: float):
    # Load the data
    data = pd.read_csv(tree_cover_data)
    data['Ground Truth Tree Cover'] = data['label'] * 10
    data['ML Tree Cover 2021'] = data['NLCD_Percent_Tree_Canopy_Cover']
    data = data.dropna(subset=['aridity_index', 'elevation', 'slope'])

    # ------------------------------------------------------------------------------
    # Subset data for within bounding region
    data_south = data[
        (data['latitude'] > min_lat) & (data['latitude'] < max_lat)
        & (data['longitude'] > min_lon) & (data['longitude'] < max_lon)
    ]

    # Convert lat/lon to radians
    data['latitude_rad'] = np.radians(data['latitude'])
    data['longitude_rad'] = np.radians(data['longitude'])

    data_south['latitude_rad'] = np.radians(data_south['latitude'])
    data_south['longitude_rad'] = np.radians(data_south['longitude'])

    # ------------------------------------------------------------------------------
    # Generate test set (S*, X*, y*) from data_south
    data_south_test = data_south.sample(frac=test_frac, random_state=42)
    S_star = data_south_test[['latitude_rad', 'longitude_rad']].values
    X_star = data_south_test[['aridity_index', 'elevation', 'slope']].values
    y_star = data_south_test['Ground Truth Tree Cover'].values

    # ------------------------------------------------------------------------------
    # Training set from data_south minus the test
    data_south_train = data_south.drop(data_south_test.index)
    S = data_south_train[['latitude_rad', 'longitude_rad']].values
    X = data_south_train[['aridity_index', 'elevation', 'slope']].values
    y = data_south_train['Ground Truth Tree Cover'].values

    # ------------------------------------------------------------------------------
    # All other data outside bounding region goes to training as well
    data_other = data[
        (data['latitude'] < min_lat) | (data['latitude'] > max_lat)
        | (data['longitude'] < min_lon) | (data['longitude'] > max_lon)
    ]
    data_other['latitude_rad'] = np.radians(data_other['latitude'])
    data_other['longitude_rad'] = np.radians(data_other['longitude'])

    S = np.concatenate([S, data_other[['latitude_rad', 'longitude_rad']].values])
    X = np.concatenate([X, data_other[['aridity_index', 'elevation', 'slope']].values])
    y = np.concatenate([y, data_other['Ground Truth Tree Cover'].values])

    # ------------------------------------------------------------------------------
    # Save arrays to disk
    datapath = Path(datadir, region)
    os.makedirs(datapath, exist_ok=True)
    np.save(Path(datapath, 'S.npy'), S)
    np.save(Path(datapath,'X.npy'), X)
    np.save(Path(datapath,'y.npy'), y)
    np.save(Path(datapath,'S_star.npy'), S_star)
    np.save(Path(datapath,'X_star.npy'), X_star)
    np.save(Path(datapath,'y_star.npy'), y_star)


if __name__ == "__main__":
    download_data()
    for region, bounds in boundary_map.items():
        save_data(region, **bounds)
    