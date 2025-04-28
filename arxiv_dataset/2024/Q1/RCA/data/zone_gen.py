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

from Classes import GpsPoint, Zone
from math import sqrt, floor


def raster(min_x, min_y, max_x, max_y, max_zones):
    """
    Generate a raster of zones, this are no zones atm
    """
    # Length of the raster
    length = max_x - min_x
    # Width of the raster
    width = max_y - min_y

    # zone count is length/side_length * width/side_length = length*width/side_length**2
    side_length = sqrt(length * width / max_zones)

    # Number of zones in x direction
    x_count = max(1, floor(length / side_length))
    x_length = length / x_count

    # Number of zones in y direction
    y_count = max(1, floor(width / side_length))
    y_length = width / y_count

    #calc the positions of zones
    positions = [GpsPoint(x * x_length + min_x, y * y_length + min_y) for x in range(x_count) for y in range(y_count)]

    #build zones
    return [Zone(pos, x_length, y_length, []) for pos in positions]



if __name__ == '__main__':
    from pprint import pprint

    r = raster(0, 0, 10, 10, 1000)
    pprint(r)
    print(len(r))
