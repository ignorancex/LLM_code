import numpy as np
import os
import argparse
from datetime import datetime


def write_wpt(lat, long, ele, name='waypoint', prefix='\t'):

    return (f'{prefix}<wpt lat="{lat}" lon="{long}">\r\n{prefix}\t<ele>{ele}</ele>'
            f'\r\n{prefix}\t<name>{name}</name>\r\n{prefix}</wpt>\r\n')


def write_trkpt(lat, long, ele, ts, ts_scale=1e-9, prefix='\t'):

    formatted_ts = datetime.fromtimestamp(int(ts * ts_scale))
    return f'{prefix}<trkpt lat="{lat}" lon="{long}"><ele>{ele}</ele><time>{formatted_ts}</time></trkpt>\r\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                Convert a fused_gps.csv file to a fused_gps.gpx GPX format (to show on online maps)
                ''')
    parser.add_argument('gps_file', help='path gps.csv file')
    parser.add_argument('--map_name', help='map name, default: My Map', default='My Map')
    parser.add_argument('--ts_scale', help='ts scaling factor [default: 1e-9 (ns)]', default=1e-9)
    args = parser.parse_args()

    gps_file = args.gps_file
    output_file = os.path.splitext(gps_file)[0] + '.gpx'
    ts_scale = np.float64(args.ts_scale)
    map_name = args.map_name

    with open(gps_file, 'r') as gps_data:

        out_data = open(output_file, 'w')

        # write header:
        out_data.write(f'<?xml version="1.0" encoding="UTF-8"?>\r\n<gpx version="1.0">\r\n\t'
                       f'<name>{map_name}</name>\r\n')

        gps_lines = gps_data.readlines()
        gps_header = gps_lines[1]
        gps_fields = gps_header.split(',')
        latitude_first = True
        if 'longitude' in gps_fields[1]:
            latitude_first = False

        first_gps_fields = gps_lines[2].split(',')
        last_gps_fields = gps_lines[-1].split(',')
        if latitude_first:
            out_data.write(write_wpt(float(first_gps_fields[1]), float(first_gps_fields[2]),
                                     float(first_gps_fields[3]), 'Start'))
            out_data.write(write_wpt(float(last_gps_fields[1]), float(last_gps_fields[2]),
                                     float(last_gps_fields[3]), 'End'))
        else:
            out_data.write(write_wpt(float(first_gps_fields[2]), float(first_gps_fields[1]),
                                     float(first_gps_fields[3]), 'Start'))
            out_data.write(write_wpt(float(last_gps_fields[2]), float(last_gps_fields[1]),
                                     float(last_gps_fields[3]), 'End'))

        out_data.write(f'\t<trk><name>{map_name}</name><number>1</number><trkseg>\r\n')

        for gps_line in gps_lines:
            if gps_line[0] == '#':
                continue
            gps_fields = gps_line.split(',')
            if latitude_first:
                out_data.write(write_trkpt(float(gps_fields[1]), float(gps_fields[2]), float(gps_fields[3]),
                            np.float64(gps_fields[0]), ts_scale, '\t\t'))
            else:
                out_data.write(write_trkpt(float(gps_fields[2]), float(gps_fields[1]), float(gps_fields[3]),
                            np.float64(gps_fields[0]), ts_scale, '\t\t'))

        out_data.write(f'\t</trkseg></trk>\r\n</gpx>')
        out_data.close()
