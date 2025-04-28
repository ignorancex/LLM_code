import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point

def main():
    # ------------------------------------------------------------------------------
    # Configure Matplotlib to use LaTeX for text rendering
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern']
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
    
    # Load the data
    datadir = os.path.join(os.path.dirname(__file__), "data")
    filename = os.path.join(datadir, "tree_cover_lin_reg_1k_ground_truth_final.csv")
    data = pd.read_csv(filename)
    # Rescale ground truth tree cover
    data["Ground Truth Tree Cover"] = data["label"] * 10
    
    # ------------------------------------------------------------------------------
    # Create a GeoDataFrame in EPSG:4326 (lat/lon)
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326"
    )

    # ------------------------------------------------------------------------------
    # Load & prepare US boundaries (reproject to EPSG:4326 as well)
    # Adjust the path below to your local .shp
    us_boundary = gpd.read_file("shape_file/s_05mr24.shp")
    us_boundary = us_boundary.to_crs("EPSG:4326")
    
    exclude_names = [
        "Alaska", "Hawaii", "Puerto Rico", "Guam", "American Samoa",
        "Northern Mariana Islands", "Fed States of Micronesia",
        "Marshall Islands", "Palau", "Virgin Islands"
    ]
    us_boundary = us_boundary[~us_boundary["NAME"].isin(exclude_names)]
    us_boundary = us_boundary.dissolve()

    # ------------------------------------------------------------------------------
    # Create figure with a 2×2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40, 21), rasterized=True)

    point_size = 40
    fontsize = 40

    # We'll define a helper to plot each panel
    def plot_map(ax, column, cmap, title):
        us_boundary.plot(ax=ax, color="white", edgecolor="black", zorder=0)
        gdf.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            markersize=point_size,
            legend=False,
            zorder=1
        )
        collection = ax.collections[-1]
        norm = collection.norm
        cbar_cmap = collection.cmap
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cbar_cmap)
        sm.set_array([])
        # Shrink the colorbar so it doesn't dominate the subplot
        cbar = plt.colorbar(sm, ax=ax, shrink=0.9)
        cbar.ax.tick_params(labelsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        # Set lat/lon bounds to roughly CONUS
        ax.set_xlim([-125, -66])
        ax.set_ylim([24, 50])
        # We allow auto aspect so lat/lon ticks appear with minimal distortion
        ax.set_aspect("auto")
        ax.set_xlabel(r"Longitude", fontsize=fontsize)
        ax.set_ylabel(r"Latitude", fontsize=fontsize)
        # set tick labels fontsize 
        ax.tick_params(axis='both', which='major', labelsize=fontsize-6)


    # ------------------------------------------------------------------------------
    # 1) Ground Truth Tree Cover
    plot_map(axes[0, 0], "Ground Truth Tree Cover", "Greens", 
             r"Target Variable (Tree Cover \%)")

    # ------------------------------------------------------------------------------
    # 2) Aridity Index
    plot_map(axes[0, 1], "aridity_index", "YlOrBr", 
             r"Aridity Index")

    # ------------------------------------------------------------------------------
    # 3) Elevation
    plot_map(axes[1, 0], "elevation", "RdPu", 
             r"Elevation (m)")

    # ------------------------------------------------------------------------------
    # 4) Slope
    plot_map(axes[1, 1], "slope", "Reds", 
             r"Slope (deg)")

    # plt.tight_layout() 
    # save the figure in rasterized format for smaller file size
    plt.savefig("tree_cover_covariates.pdf", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
