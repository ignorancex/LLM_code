"""
Copyright (C) 2024 Tom Mucke, Finn Seesemann
Ideas and Theory by Felix Engelhardt, Tom Mucke, Alexander Renneke, Finn Seesemann

The data read by this program has its own copyright, please refer to data/tidal_points/COPYRIGHT.

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
import csv
from datetime import datetime, timedelta

nortern_path = "data/tidal_points/north_sea.csv"
baltic_path = "data/tidal_points/baltic_sea.csv"
date_range_path = "data/tidal_points/date_range.txt"


def parse_tidal_points(path: str):
    # Log COPYRIGHT to console
    print("The data read by this program has its own copyright,"
          " please refer to data/tidal_points/COPYRIGHT.")

    path = os.path.normpath(path)
    target_dir = path.removesuffix(".csv")
    os.makedirs(target_dir, exist_ok=True)

    skip_count = 0

    with open(date_range_path, "r") as f:
        start = datetime.strptime(f.readline().strip(), "%Y-%m-%d")
        end = datetime.strptime(f.readline().strip(), "%Y-%m-%d") + timedelta(days=1)

    with open(path, "r") as f:
        f.readline()  # skip Copyright
        reader = csv.reader(f, delimiter=";")
        current_date = start
        current_level = None
        current_correction = 0
        current_target_file = None
        current_target_writer = None
        prev_time = start
        try:
            for row in reader:
                if len(row) == 0:
                    continue
                if len(row) > 2:
                    assert row[10] == "PNP", "unexpected base niveau"
                    old_level = current_level
                    old_date = current_date
                    current_date = datetime.strptime(row[0], "%d.%m.%Y")
                    current_level = str(row[4])
                    current_correction = float(row[-1].replace(",", ".")) * 100  # m to cm
                    if old_level != current_level:
                        if end - prev_time != timedelta(minutes=0) and old_level is not None:
                            logging.debug(f"Missing {int((end - prev_time).total_seconds() / 60)} entries from "
                                            f"{prev_time.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}"
                                            f" for {old_level} in {path}")
                            for i in pd.date_range(prev_time + timedelta(minutes=1), end, freq="min"):
                                current_target_writer.writerow([i.strftime("%Y-%m-%d|%H:%M"), ""])
                                skip_count += 1
                        if current_target_file is not None:
                            current_target_file.close()
                        target_file_path = os.path.join(target_dir, current_level + ".csv")
                        if os.path.exists(target_file_path):
                            logging.debug("Overriding " + target_file_path)
                        current_target_file = open(target_file_path, "w", newline='')
                        current_target_writer = csv.writer(current_target_file)
                        current_target_writer.writerow(["date_time", "NHN"])
                        old_date = start
                    continue
                else:
                    assert current_target_writer is not None
                    if row[0] == "24:00":
                        t = current_date + timedelta(days=1)
                    else:
                        t = datetime.strptime(row[0], "%H:%M")
                        t = t.replace(year=current_date.year, month=current_date.month, day=current_date.day)
                    if t - prev_time > timedelta(minutes=1):
                        logging.debug(f"Missing {int((t - prev_time).total_seconds() / 60)} entries from "
                                        f"{prev_time.strftime('%Y-%m-%d %H:%M')} to {t.strftime('%Y-%m-%d %H:%M')} for "
                                        f"{current_level} in {path}")
                        for i in pd.date_range(prev_time + timedelta(minutes=1), t, freq="min"):
                            skip_count += 1
                            current_target_writer.writerow([i.strftime("%Y-%m-%d|%H:%M"), ""])
                    prev_time = t
                    if row[1] == "XXX,XXX":
                        logging.debug("Skipping " + t.strftime("%Y-%m-%d|%H:%M") + " because of XXX,XXX")
                        current_target_writer.writerow([t.strftime("%Y-%m-%d|%H:%M"), ""])
                        skip_count += 1
                    else:
                        current_target_writer.writerow(
                            [t.strftime("%Y-%m-%d|%H:%M"), float(row[1].replace(",", ".")) + current_correction])
        finally:
            if current_target_file is not None:
                current_target_file.close()

    if skip_count > 0:
        logging.warning("Skipped " + str(skip_count) + " entries in " + path)
    return skip_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    skip_count = 0
    skip_count += parse_tidal_points(nortern_path)
    skip_count += parse_tidal_points(baltic_path)
    if skip_count > 0:
        logging.error("Total Skipped " + str(skip_count) + " entries")
