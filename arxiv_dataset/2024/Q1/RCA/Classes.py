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

from __future__ import annotations

import json
import logging
from itertools import count
from json.encoder import INFINITY, _make_iterencode, encode_basestring_ascii, encode_basestring

from haversine import haversine, Unit
from functools import cached_property
from functools import lru_cache

from enum import Enum, auto


class GpsPoint:
    def __init__(self, latitude, longitude):
        self.latitude: float = latitude
        self.longitude: float = longitude

    def __str__(self):
        return f"GpsPoint with latitude {self.latitude} and longitude {self.longitude}"

    def __repr__(self):
        return f"{self.latitude}; {self.longitude}"

    def __eq__(self, other):
        assert isinstance(other, GpsPoint)
        return self.latitude == other.latitude and self.longitude == other.longitude

    def __hash__(self):
        return hash((self.latitude, self.longitude))

    def distance_by_air(self, other):
        return haversine((self.latitude, self.longitude), (other.latitude, other.longitude), unit=Unit.NAUTICAL_MILES)

    @lru_cache(maxsize=128)
    def distance_to(self, other):
        return self.distance_by_air(other)

    def from_google_maps_link(link: str) -> GpsPoint:
        parts = link.split("=")
        coords = parts[-1].split(",")
        return GpsPoint(latitude=float(coords[0]), longitude=float(coords[1]))


class VesselType:
    _counter = count()

    def __init__(self, amount, speed, reach=0, tools=None, draught=None, identifier=None):
        """

        :param amount: Number of available vessels of this type
        :param speed: Speed of the vessel in knots
        """
        if identifier is None:
            self.identifier = next(self._counter)
        else:
            self.identifier = identifier
            self._counter = count(max(next(self._counter), self.identifier + 1))
        self.amount = amount
        self.speed = speed
        self.allowed_ports = set()
        self.reach = reach
        self.tools = tools
        if self.tools is None:
            self.tools = {}
        self.draught = draught

    def __str__(self):
        return f"VesselType {self.identifier} with speed {self.speed} and amount {self.amount}"

    def __repr__(self):
        return f"VesselType {self.identifier}; {self.speed}; {self.amount}"

    def __eq__(self, other):
        assert isinstance(other, VesselType)
        return self.identifier == other.identifier

    def __hash__(self):
        return self.identifier


class UniqueVesselType(VesselType):

    def __eq__(self, other):
        assert isinstance(other, VesselType)
        return self.speed == other.speed

    @classmethod
    def from_json(cls, json_str, amount=1, additional_info=None):
        speed_str: str = json_str["Geschwindigkeit"]
        speed_str = speed_str.split(" Knoten")[0].replace(",", ".")
        draught = json_str.get("Tiefgang", None)
        if draught is None:
            logging.warning(f"No draught for {json_str['Name']} found")
        else:
            draught = float(draught.removesuffix(" Meter").removesuffix(" m").replace(",", "."))
        reach = -1
        tools = {}
        if additional_info is not None:
            amount = additional_info.get("amount", amount)
            reach = additional_info.get("range", [{
                "speed": -1,
                "range": -1
            }])
            reach = max(reach, key=lambda x: x["range"])["range"]
            tools = additional_info.get("tools", {})
        return cls(amount=amount, speed=float(speed_str), reach=reach, tools=tools, draught=draught)

    def __repr__(self):
        return f"UniqueVesselType {self.speed}; {self.amount}"

    @cached_property
    def vessel_type(self) -> VesselType:
        return VesselType(self.amount, self.speed, self.reach, self.tools, self.draught, self.identifier)


class Station:
    _counter = count()

    def __init__(self, minimum_water_level: float,
                 allowed_vessels: list[VesselType], position: GpsPoint,
                 name: str = "", callsign: str = "", depth=None, identifier=None):
        if identifier is None:
            self.identifier = next(self._counter)
        else:
            self.identifier = identifier
            self._counter = count(max(next(self._counter), self.identifier + 1))
        self.allowed_vessels = set(allowed_vessels)
        for vessel_type in self.allowed_vessels:
            vessel_type.allowed_ports.add(self)
        self.position = position
        self.minimum_water_level = minimum_water_level
        self.name = name
        self.callsign = callsign
        self.depth = depth

    def __str__(self):
        return f"Port {self.identifier} at {self.position} with name {self.name}"

    def __repr__(self):
        return f"Port {self.identifier}; {self.callsign}; {len(self.allowed_vessels)};"

    def __eq__(self, other):
        assert isinstance(other, Station)
        return self.identifier == other.identifier

    def __hash__(self):
        return self.identifier

    def add_allowed_vessel(self, vessel: VesselType) -> None:
        self.allowed_vessels.add(vessel)
        vessel.allowed_ports.add(self)


class Zone:
    _counter = count()

    def __init__(self, position: GpsPoint, width: float = 0, height: float = 0,
                 reachable_from_by: list[tuple[Station, VesselType]] = None, identifier=None):
        if reachable_from_by is None:
            reachable_from_by = []
        if identifier is None:
            self.identifier = next(self._counter)
        else:
            self.identifier = identifier
            self._counter = count(max(next(self._counter), self.identifier + 1))
        self.position = position
        self.width = width
        self.height = height
        self.reachable_from_by = set(reachable_from_by)

    def __str__(self):
        return f"Zone at {self.position} with width {self.width} height {self.height}"

    def __repr__(self):
        return f"Zone {self.position}; {self.width}; {self.height}"

    def __eq__(self, other):
        assert isinstance(other, Zone)
        return self.position == other.position and self.width == other.width and self.height == other.height

    def __hash__(self):
        return hash((self.position, self.width, self.height))

    def addReachableFromBy(self, station: Station, vesseltype: VesselType) -> None:
        self.reachable_from_by.add((station, vesseltype))


class Incident_Type:
    _counter = count()

    def __init__(self, allowed_vessels: list[VesselType], probability_by_zone: dict[Zone, float], weight: float,
                 identifier=None):
        if identifier is None:
            self.identifier = next(self._counter)
        else:
            self.identifier = identifier
            self._counter = count(max(next(self._counter), self.identifier + 1))
        self.allowed_vessels = set(allowed_vessels)
        self.probability_by_zone = probability_by_zone
        self.weight = weight

    def __eq__(self, other):
        assert isinstance(other, Incident_Type)
        return self.identifier == other.identifier

    def __hash__(self):
        return self.identifier


class Water(Enum):
    NORTH_SEA = 1
    BALTIC_SEA = 2
    ALL = 3

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value:
                    return member
            if value in ("NORTHERNSEA", "NORTHSEA", "NORDSEE"):
                return cls.NORTH_SEA
            elif value in ("BALTICSEA", "OSTSEE"):
                return cls.BALTIC_SEA


class _Dummy:
    pass


class SolveType(Enum):
    GUROBI_MANY_ZONES = auto()
    GUROBI_BETTER_TIDAL = auto()
    GUROBI_BEST_TIDAL = auto()


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, GpsPoint):
            return {"__dtype__": "GpsPoint", "latitude": o.latitude, "longitude": o.longitude}
        elif isinstance(o, VesselType):
            return {"__dtype__": "Vessel", "amount": o.amount, "speed": o.speed, "identifier": o.identifier,
                    "reach": o.reach,
                    "tools": o.tools, "draught": o.draught}
        elif isinstance(o, Station):
            return {"__dtype__": "Station", "identifier": o.identifier,
                    "allowed_vessels": list(o.allowed_vessels), "position": o.position, "name": o.name,
                    "callsign": o.callsign, "depth": o.depth, "minimum_water_level": o.minimum_water_level}
        elif isinstance(o, Zone):
            return {"__dtype__": "Zone", "identifier": o.identifier, "position": o.position, "width": o.width,
                    "height": o.height,
                    "reachable_from_by": list(o.reachable_from_by)}
        elif isinstance(o, Incident_Type):
            return {"__dtype__": "Incident_Type", "identifier": o.identifier,
                    "allowed_vessels": list(o.allowed_vessels),
                    "probability_by_zone": o.probability_by_zone, "weight": o.weight}
        elif isinstance(o, SolveType):
            return {"__dtype__": "SolveType", "name": o.name}
        elif isinstance(o, Water):
            return {"__dtype__": "Water", "name": o.name}
        elif isinstance(o, tuple):
            return {"__dtype__": "tuple", "value": list(o)}
        return super().default(o)

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(o, allow_nan=self.allow_nan,
                     _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        assert not _one_shot, "Not Implemented"
        _iterencode = _make_iterencode(
            markers, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot, tuple=_Dummy)
        return _iterencode(o, 0)


def json_decoder_hook(dct):
    if "__dtype__" in dct:
        if dct["__dtype__"] == "Vessel":
            return VesselType(amount=dct["amount"], speed=dct["speed"], identifier=dct["identifier"],
                              reach=dct["reach"], tools=dct["tools"], draught=dct["draught"])
        elif dct["__dtype__"] == "Station":
            return Station(identifier=dct["identifier"],
                           allowed_vessels=dct["allowed_vessels"], position=dct["position"], name=dct["name"],
                           callsign=dct["callsign"], depth=dct["depth"], minimum_water_level=dct["minimum_water_level"])
        elif dct["__dtype__"] == "Zone":
            return Zone(position=dct["position"], width=dct["width"], height=dct["height"],
                        reachable_from_by=dct["reachable_from_by"], identifier=dct["identifier"])
        elif dct["__dtype__"] == "Incident_Type":
            return Incident_Type(identifier=dct["identifier"], allowed_vessels=dct["allowed_vessels"],
                                 probability_by_zone=dct["probability_by_zone"], weight=dct["weight"])
        elif dct["__dtype__"] == "GpsPoint":
            return GpsPoint(latitude=dct["latitude"], longitude=dct["longitude"])
        elif dct["__dtype__"] == "tuple":
            return tuple(dct["value"])
        elif dct["__dtype__"] == "SolveType":
            return SolveType[dct["name"]]
        elif dct["__dtype__"] == "Water":
            return Water[dct["name"]]
    return dct
