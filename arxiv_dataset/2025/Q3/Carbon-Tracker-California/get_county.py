import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Step 1: Load the shapefile containing county and state boundaries
# Replace with the path to your shapefile of counties (with state information)
shapefile_path = "data\CA_Counties\CA_Counties.shp"  # Update this with the actual file path
county_gdf = gpd.read_file(shapefile_path)
# Step 2: Load the CSV file with longitude and latitude
input_csv = "data/lat_lon_data.csv"  # Replace with your input CSV file
coordinates_df = pd.read_csv(input_csv)

# Step 3: Convert the CSV data to a GeoDataFrame
geometry = [Point(xy) for xy in zip(coordinates_df['Lon'], coordinates_df['Lat'])]
geo_df = gpd.GeoDataFrame(coordinates_df, geometry=geometry, crs="EPSG:4326")

# Step 4: Perform a spatial join with the county GeoDataFrame
# Ensure both GeoDataFrames are in the same CRS
county_gdf = county_gdf.to_crs("EPSG:4326")
result_gdf = gpd.sjoin(geo_df, county_gdf, how="left", predicate="intersects")

# Step 5: Save the output to a new CSV file
# `county_name` and `state_name` are assumed to be the relevant columns in your shapefile
output_csv = "output_with_county_state.csv"
result_df = result_gdf[['Lon', 'Lat', 'NAMELSAD']]
result_df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
