import pandas as pd
import numpy as np

import logging 
import geojson
import os

import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import shapely.ops
from pyproj import Proj

from bs4 import BeautifulSoup
import requests
import us

logging.basicConfig(level=logging.WARNING)

with os.popen("git rev-parse --show-toplevel") as path:
    my_path = path.read().strip("\n")

def load_county_hesitancy_data(county = None, state = None, download_data = False):
    """ Return geodataframe with hesitancy data.

    Inputs:
        county: (str) Capitalized county name, or None.
        state: (str) Uppercase two-letter state name abbreviation, or None.
        download_dada: (bool) if True downloads data from CDC.

    Returns: 
        Single row dataframe with hesitancy estimates for CDC week 31.
    """
    full_state_name = us.states.lookup(state).name
    county = county.lower().split(" county")[0]

    if download_data == False:
        hesitancy_df = pd.read_csv(
            "{}/data/{}_county_{}_hesitancy.csv".format(
            my_path, county.lower(), state.lower()), index_col = 0)
    else:
        try:
            hesitancy_df = pd.read_csv(
                "https://data.cdc.gov/api/views/q9mh-h2tw/rows.csv?accessType=DOWNLOAD")
            hesitancy_df[hesitancy_df["County Name"] == "{} County, {}".format(
                            county.capitalize(), full_state_name.capitalize())]
        except:
            raise ValueError("It looks like the cdc url is no longer active.  You can"+
                "access an archived copy of the data at https://tinyurl.com/ODyNhesitancy")

    return hesitancy_df

def get_county_mapping_data(county = None, state = None):
    """ Return geodataframe with location data.

    Inputs:
        county: (str) Capitalized county name, or None.
        state: (str) Uppercase two-letter state name abbreviation, or None.
        
    Returns: 
        Geodataframe with polygon geometry, population and 
        area data for county, state.
    """
    df = pd.DataFrame(index = [0], columns = ["county","state","area (km^2)",
                    "population (2019)", "density","geometry"])
    if county is None:
        county = "Montgomery"
    df.loc[0,"county"] = county

    if state is None:
        state = "AL"
    df.loc[0,"state"] = state

    county = county.split(" County")[0].split(" county")[0].capitalize()
    state = state.upper()

    geolocator = Nominatim(user_agent="VaccineHesitancy")
    geo = geolocator.geocode("{} County, {}".format(county, state),
                            geometry='geojson')

    if geo is None:
        raise ValueError("Check the spelling of your county name.")

    assert geo.raw["display_name"].split(", ")[0] == "{} County".format(
                    county), "Chcek the spelling of your county name."

    full_state_name = geo.raw["display_name"].split(", ")[1]
    polygon = Polygon(geo.raw["geojson"]["coordinates"][0])
    df["geometry"] = polygon

    # Convert to GeoDataFrame
    geo_df = gpd.GeoDataFrame(df, index=[0], crs='epsg:4326', geometry=[polygon])

    #Add spherical coordinate reference system (crs) to lat/long pairs.
    geo_df.crs = "EPSG:4326" 

    #Project onto a flat crs for mapping.
    geo_df = geo_df.to_crs(epsg=3857) 

    # Get area
    lon, lat = zip(*geo.raw["geojson"]["coordinates"][0])
    pa = Proj(
        "+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55")
    x, y = pa(lon, lat)
    coord_proj = {"type": "Polygon", "coordinates": [zip(x, y)]}
    area = shape(coord_proj).area/(10**6) # area in km^2

    geo_df["area (km^2)"] = area

    # Get population.

    url = "https://en.wikipedia.org/wiki/List_of_United_States_counties_and_county_equivalents"

    # Load county population 
    request = requests.get(url) 
    html_content = request.text
    soup = BeautifulSoup(html_content,"lxml")
    county_tables = soup.find_all("table", {"class":"wikitable sortable"})

    df=pd.read_html(str(county_tables))
    df=pd.DataFrame(df[0])
    df["County or equivalent"] = [g.split("[")[0] for g in df["County or equivalent"]]
    df = df[(df["County or equivalent"] == county)&(df["State or equivalent"
        ] == full_state_name)]

    geo_df["population (2020)"] = df["Population (2020 census)"].iloc[0]
    geo_df["density"] = geo_df["population (2020)"]/geo_df["area (km^2)"]

    # Load vaccine hesitancy data.
    try: 
        hesitancy_df = load_county_hesitancy_data(county = county, 
                                                state = state, 
                                                download_data = False)

    except:
        hesitancy_df = load_county_hesitancy_data(county = county, 
                                                state = state, 
                                                download_data = True)

    geo_df["strongly_hesitant"] = hesitancy_df[
                    "Estimated strongly hesitant"].iloc[0]
    geo_df["hesitant_or_unsure"] = hesitancy_df[
                    "Estimated hesitant or unsure"].iloc[0] - geo_df["strongly_hesitant"]
    geo_df["not_hesitant"] = 1 - hesitancy_df[
                    "Estimated hesitant or unsure"].iloc[0]

    return geo_df

def get_hesitancy_dict(geo_df):
    """ Return dictionary of hesitancy proportions.

    Input:
        geo_df: (dataframe) location geomatic dataframe typically output 
            from get_county_mapping_data(). 

    Output: 
        Dictionary with keys not_hesitant, hesitant_or_unsure, and 
        strongly_hesitant.
    """
    return {"not_hesitant" : geo_df.loc[0,"not_hesitant"],
            "hesitant_or_unsure" : geo_df.loc[0,"hesitant_or_unsure"],
            "strongly_hesitant" : geo_df.loc[0,"strongly_hesitant"]}


def make_triangulation(geo_df):
    """ Print geojson dictionary with county triangulation.

    Input:
        geo_df: (dataframe) location geomatic dataframe typically output 
            from get_county_mapping_data(). 

    Output: 
        Geojson file printed to ../data/<state abbreviation>/<county>
    """
    county = geo_df.loc[0,"county"].lower()
    state = geo_df.loc[0,"state"].lower()
    path = "../data/{}".format(state)
    if os.path.exists(path) == False:
        os.mkdir(path)
    path = path + "/{}".format(county)
    if os.path.exists(path) == False:
        os.mkdir(path)

    tri = shapely.ops.triangulate(geo_df.loc[0,"geometry"])
    inner_tri = [t for t in tri if t.within(geo_df.loc[0,"geometry"])]

    tri_dict = {"type":"Feature",
                    "geometry": {
                        "type":"MultiPolygon",
                        "coordinates":[list(t.exterior.coords
                            ) for t in inner_tri]
                                },
                "properties":county}
    label = county.split(", {}".format(state))[0].replace(" ","_").lower()

    with open('{}/triangulation_dict.geojson'.format(path), 'w') as the_file:
        geojson.dump(tri_dict, the_file)
    return tri_dict