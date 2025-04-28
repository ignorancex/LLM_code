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

import logging
import os

import pandas as pd
from sklearn.cluster import KMeans

from Classes import VesselType, Station, Zone, Incident_Type, Water, GpsPoint
import data.zone_gen as zone_gen
from random import random, randint

from tidal_gen.generate_intervals import consolidate_intervals


class DataHouse:
    def __init__(self, water: Water):
        self.vessel_types: list[VesselType]
        self.stations: list[Station]
        self.zones: list[Zone]
        self.zones_clustered = list[Zone]
        self.incident_types: list[Incident_Type]
        self.shares: dict[tuple[Station, VesselType], float]
        self.water = water

    def create_all(self, max_zones=100, seed: int = None, consolidate=True) -> None:
        logging.info("Creating DataHouse for " + str(self.water))
        self.fill_in(consolidate=consolidate)
        logging.info("Creating vellel station allowances")
        self.random_allowed_vessels_to_stations(p=0.9)

        if seed is None:
            logging.critical("Creating random zones since no seed was given. These zones will likely result in an"
                             "infeasible problem, please create zones using the zone_gen.py script")
            self.random_zones(max_zones=max_zones, oversize=0.2)
        else:
            logging.info("Loading zones from file")
            self.load_zones(seed=seed, number_of_zones=max_zones)
            logging.info("Done loading zones, loaded " + str(len(self.zones)) + " zones")

        logging.info("Creating reachable tuples")
        self.reachable_for_zones()
        logging.info("Creating incident types")
        self.random_incident_types(tow_amount=10, vessel_p=0.6, zone_p=0.4)
        logging.info("Done creating DataHouse")

    def get_random_bool(self, p: float) -> bool:
        value = randint(0, 100)
        return value < p * 100

    def get_random_p(self, min: float = 0, max: float = 1):
        value = random()
        scaled = min + (value * (max - min))
        return round(scaled, 4)

    # get the different types out of the raw file
    def fill_in(self, consolidate: bool) -> None:
        logging.info("Filling in data...")
        if consolidate:
            self.shares, self.stations, self.vessel_types = consolidate_intervals(self.water)
        else:
            self.unconsolidated_shares, self.shares, self.stations, self.vessel_types = consolidate_intervals(
                self.water, False)
        logging.info("Done filling in data")

    def random_allowed_vessels_to_stations(self, p: float = 0.9) -> None:
        values = [[self.get_random_bool(p) for _ in self.vessel_types] for _ in self.stations]

        # go over all stations
        for i, station in enumerate(self.stations):
            for j, vessel in enumerate(self.vessel_types):
                if values[i][j]:
                    station.add_allowed_vessel(vessel)

        # validate (no station without allowed vessel)
        for station in self.stations:
            if len(station.allowed_vessels) == 0:
                logging.warning("There was a station without allowed vessel... adding first vesseltype as default")
                station.add_allowed_vessel(self.vessel_types[0])

        # remove unallowed combinations from shares
        self.shares = dict(filter(lambda share: share[0][0] in share[0][1].allowed_ports, self.shares.items()))

    # oversize means, that the border of zone area is 20% of total size (border means, build a rectangle on all station position)
    def random_zones(self, oversize: float = 0.2, max_zones=100) -> None:
        # first calc the edge points
        all_lat = [station.position.latitude for station in self.stations]
        all_long = [station.position.longitude for station in self.stations]
        min_lat = min(all_lat)
        max_lat = max(all_lat)
        min_long = min(all_long)
        max_long = max(all_long)
        # calc sizes
        lat_size = max_lat - min_lat
        long_size = max_long - min_long

        # now build oversize
        min_lat -= lat_size * oversize
        max_lat += lat_size * oversize
        min_long -= long_size * oversize
        max_long += long_size * oversize

        # now create zones
        self.zones = zone_gen.raster(min_x=min_long, max_x=max_long, min_y=min_lat, max_y=max_lat, max_zones=max_zones)

    def load_zones(self, seed: int, number_of_zones: int) -> None:
        match self.water:
            case Water.NORTH_SEA:
                foldername = "northern_sea"
            case Water.BALTIC_SEA:
                foldername = "eastern_sea"
            case _:
                foldername = "combined"

        folder = os.path.join("data/geo", foldername)

        def create_zone_from_row(row) -> Zone:
            gps_point = GpsPoint(longitude=row["lon"], latitude=row["lat"])
            return Zone(gps_point)

        with open(os.path.join(folder, str(number_of_zones) + "_" + str(seed) + ".csv"), "r") as f:
            df = pd.read_csv(f, sep=",")

        self.zones = df.apply(create_zone_from_row, axis=1).tolist()

    # fill in reachable tuples
    def reachable_for_zones(self, zones=None) -> None:
        # values = [[[self.get_random_bool(p) for _ in self.vessel_types] for _ in self.stations] for _ in self.zones]
        if zones is None:
            zones = self.zones
        # go over all zones
        for idx, zone in enumerate(zones):
            for i, station in enumerate(self.stations):
                for j, vessel in enumerate(self.vessel_types):
                    if zone.position.distance_to(station.position) <= vessel.reach / 2:
                        zone.addReachableFromBy(station, vessel)

    # create incident types
    def random_incident_types(self, tow_amount: int = 10, vessel_p: float = 0.6, zone_p: float = 0.4) -> None:
        # save the weights first

        boolean_incident_types = [
            "firefighting",
            "pumping",
            "second_craft",
            "first_aid",
            "board_hospital"]
        weights = [self.get_random_p() for _ in boolean_incident_types]
        result = []
        for i, incident_type in enumerate(boolean_incident_types):
            # create allowed vessels
            allowed_vessels = list()
            for idx, vessel in enumerate(self.vessel_types):
                if vessel.tools.get(incident_type, False):
                    allowed_vessels.append(vessel)

            assert len(allowed_vessels) > 0, "There was no vessel with " + incident_type + " tools"

            # create prop by zone
            prop_by_zone = dict()
            # first decide, if incident should appear in the zone
            appear_bools = [self.get_random_bool(zone_p) for _ in self.zones]
            for idx, zone in enumerate(self.zones):
                if appear_bools[idx]:
                    prop_by_zone[zone] = self.get_random_p()
                else:
                    prop_by_zone[zone] = 0

            # now build incident obj
            incident = Incident_Type(allowed_vessels, prop_by_zone, weights[i])
            result.append(incident)

        weights = [self.get_random_p() for _ in range(tow_amount)]
        max_tow_weight = max([vessel.tools.get("towing", 0) for vessel in self.vessel_types])
        for tow in range(tow_amount):
            tow_weight = self.get_random_p(0, max_tow_weight)
            # create allowed vessels
            allowed_vessels = list()
            for idx, vessel in enumerate(self.vessel_types):
                if vessel.tools.get("towing", 0) >= tow_weight:
                    allowed_vessels.append(vessel)

            assert len(allowed_vessels) > 0, "There was no vessel with tow tools"

            # create prop by zone
            prop_by_zone = dict()
            # first decide, if incident should appear in the zone
            appear_bools = [self.get_random_bool(zone_p) for _ in self.zones]
            for idx, zone in enumerate(self.zones):
                if appear_bools[idx]:
                    prop_by_zone[zone] = self.get_random_p()
                else:
                    prop_by_zone[zone] = 0
            incident = Incident_Type(allowed_vessels, prop_by_zone, weights[tow] / tow_amount)
            result.append(incident)
        # store result
        self.incident_types = result

    # clusters the zones and updates the incident types
    def cluster_zones(self, final_amount: int):
        # calc the kmeans cluster
        x_coords = [zone.position.longitude for zone in self.zones]
        y_coords = [zone.position.latitude for zone in self.zones]
        df_coords = pd.DataFrame({"x": x_coords, "y": y_coords})
        kmeans = KMeans(n_clusters=final_amount, random_state=0, n_init="auto").fit(df_coords)

        # track which new zone (by id) has which zones as origin
        origin_zones = [[] for _ in range(final_amount)]
        for idx in range(len(self.zones)):
            old_zone = self.zones[idx]
            new_zone_idx = kmeans.labels_[idx]
            origin_zones[new_zone_idx].append(old_zone)
        # now create new zone objects with values
        new_zones = list()
        new_incident_prop_by_zone = {incident: dict() for incident in self.incident_types}
        for idx in range(final_amount):
            old_zones = origin_zones[idx]
            coord = kmeans.cluster_centers_[idx]
            # create new zone
            gps = GpsPoint(longitude=coord[0], latitude=coord[1])
            zone = Zone(gps)
            new_zones.append(zone)
            # now calc new incident prop by zone
            for incident in self.incident_types:
                new_prop = sum([incident.probability_by_zone[z] for z in old_zones]) / len(old_zones)
                new_incident_prop_by_zone[incident][zone] = new_prop
        self.zones_clustered = new_zones  # store the zones

        # assign new values to incident types
        for incident in self.incident_types:
            incident.probability_by_zone = new_incident_prop_by_zone[incident]

        # calc the reachable from by stuff
        self.reachable_for_zones(self.zones_clustered)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dh = DataHouse(Water.ALL)
    dh.create_all(100, 10)
    dh.cluster_zones(10)
