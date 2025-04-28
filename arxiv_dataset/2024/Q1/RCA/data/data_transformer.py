"""
Copyright (C) 2024 Tom Mucke, Finn Seesemann
Ideas and Theory by Felix Engelhardt, Tom Mucke, Alexander Renneke, Finn Seesemann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from Classes import VesselType, UniqueVesselType, Station, GpsPoint
import json

raw_data_path = "./data/raw/"


def transform_vesseltypes() -> list[VesselType]:
    unique_field = "further_link"

    # load data from file
    raw_data = None
    with open(raw_data_path + "raw_data_with_depth.json", "r") as f:
        raw_data = json.load(f)
    ship_info = [data["ship_info"] for data in raw_data]

    with open(raw_data_path + "vessels.json", "r") as f:
        vessel_additional_info = json.load(f)

    # remove empty elements (for example bremen is a station without a ship)
    ship_info = list(filter(lambda ship: "ship_class" in ship, ship_info))
    print("Amount of vessels: " + str(len(ship_info)))

    # filter unique ship classes
    ship_classes = dict()
    for ship in ship_info:
        ship_classes[ship[unique_field]] = ship_classes.get(ship[unique_field], [])
        ship_classes.get(ship[unique_field]).append(ship)
    # sort the values (just for overview)
    ship_classes_l = list(sorted(ship_classes.values(), key=lambda ship: ship[0]["ship_class"][0:2]))
    print("Amount of unique vessels: " + str(len(ship_classes)))

    # create objects for the data types
    ship_objs = dict()
    for ship_key in ship_classes:
        ship_objs[ship_key] = []
        for ship in ship_classes[ship_key]:
            ship_objs[ship_key].append(UniqueVesselType.from_json(ship, additional_info=vessel_additional_info[ship_key]))

    # validate the data of the ships in one class
    result = []
    for ship_key in ship_objs:
        # always add first element
        arr = ship_objs[ship_key]
        unique_obj = dict()
        for ship in arr:
            if ship.__repr__() in unique_obj:
                unique_obj[ship.__repr__()].amount += 1
            else:
                unique_obj[ship.__repr__()] = ship
        result += unique_obj.values()

    print("Amount of real unique vessels: " + str(len(result)))
    return [unique_vessel_type.vessel_type for unique_vessel_type in result]


def transform_stations() -> list[Station]:
    # load data from file
    raw_data = None
    with open(raw_data_path + "raw_data_with_depth.json", "r") as f:
        raw_data = json.load(f)
    # filter station without harbour
    data_filtered = list(filter(lambda data: len(data["ship_info"].values()) > 0, raw_data))

    # extract stations
    result = []
    for data in data_filtered:
        # read out subinfo
        google_maps_link = data["basic_info"]["location"]
        station_info = data["station_info"]
        # callsign
        callsign = ""
        if "Ruf\xadzeichen" in station_info:
            callsign = station_info["Ruf\xadzeichen"]
        elif "Rufzeichen" in station_info:
            callsign = station_info["Rufzeichen"]
        else:
            print("No callsign available for " + station_info["station_name"])
        callsign = callsign.replace(" ", "_")  # replace blank with _ for repr it is better for readablility

        # get the coords for the station
        coords = GpsPoint.from_google_maps_link(google_maps_link)
        # create and store station object
        result.append(Station(0, [], coords, name=station_info["station_name"], callsign=callsign,
                              depth=station_info["calculated_depth"]))

    print("Amount of stations: " + str(len(result)))
    return result
