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

import json
import sys
import os
import logging
import importlib
import importlib.util
from importlib.abc import MetaPathFinder
from collections import deque


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
"""
Note: Requires to use the PyQuis Python Environment
"""
QGISPATH = os.path.normpath("C:/Program Files/QGIS 3.34.1/apps/qgis")  # Change this to your QGIS installation path
DATAFOLDER = "data/geo"
logging.info("QGISPATH: " + QGISPATH)

# ensure that PyQuis Python Environment is used
logging.info("Python executable: " + sys.executable)
parent_QGISPATH = os.path.normpath(os.path.join(QGISPATH, ".."))
assert os.path.normpath(sys.executable).startswith(parent_QGISPATH), "Please use the PyQuis Python Environment"

# add qgis python path to sys.path
sys.path.append(QGISPATH)
sys.path.append(os.path.join(QGISPATH, "python"))
sys.path.append(os.path.join(QGISPATH, "python/plugins"))
sys.path.append(os.path.join(QGISPATH, "share/qgis/python"))
sys.path.append(os.path.join(QGISPATH, "Contents/Resources/python"))


# create a pathfinder for processing because qgis thought it might be funny to name the module after an existing one...
class ProcessingPathFinder(MetaPathFinder):
    @classmethod
    def find_spec(cls, name, path, target=None):
        if name.startswith("processing"):
            # use importlib to create a spec for the module from the plugins folder
            file = os.path.join(QGISPATH, "python/plugins", name.replace('.', '/'), '__init__' ".py")
            if not os.path.exists(file):
                file = os.path.join(QGISPATH, "python/plugins", name.replace('.', '/') + ".py")
            if not os.path.exists(file):
                return None
            spec = importlib.util.spec_from_file_location(name, file)
            return spec
        elif name.startswith("haversine"):
            # Classes imports haversine, but we will not use it so we will point to a dummy module
            spec = importlib.util.spec_from_file_location(name, "tidal_gen/__dummy__.py")
            return spec


sys.meta_path.insert(0, ProcessingPathFinder)

# now we can finally import qgis
from qgis.core import *
from qgis.PyQt.QtCore import QVariant


class LoggingFeedback(QgsProcessingFeedback):
    def __init__(self):
        super().__init__()
        self.progressChanged.connect(self.on_progress_changed)

    def on_progress_changed(self, progress):
        logging.debug(f"Progress: {progress}")
        sys.stdout.write("\rProgress: {0:.2f}%".format(progress))
        sys.stdout.flush()

    def reportError(self, error, fatalError=False):
        if fatalError:
            logging.critical(error)
        else:
            logging.error(error)

    def pushInfo(self, info):
        logging.info(info)

    def pushCommandInfo(self, info):
        logging.info(info)

    def pushDebugInfo(self, info):
        logging.debug(info)


def get_depth_at_point(layer: QgsRasterLayer, point: QgsPointXY):
    RESOLUTION = 0.0005
    stack = deque()
    stack.append(point)
    added = {(point.x(), point.y())}
    while len(stack) > 0:
        if len(added) > 35:
            logging.debug("Too far from start point, switching to -10")
            return -10  # these points are most likely not properly covered by the raster or are land based stations
        point = stack.popleft()
        values = layer.dataProvider().identify(point, QgsRaster.IdentifyFormatFeature)
        assert values.isValid()
        z = ((values.results())[0][0]).features()[0].attributes()[0]
        if z != 999999:
            logging.debug(f"Length of added: {len(added)}")
            return z
        # add neighbours
        x, y = point.x(), point.y()
        for z in (x + RESOLUTION, y), (x - RESOLUTION, y), (x, y + RESOLUTION), (x, y - RESOLUTION):
            if z not in added:
                stack.append(QgsPointXY(*z))
                added.add(z)


def create_layers():
    # supply path to qgis install location
    QgsApplication.setPrefixPath(QGISPATH, True)

    # create a reference to the QgsApplication, setting the
    # second argument to False disables the GUI
    qgs = QgsApplication([], False)

    # load providers
    qgs.initQgis()
    import processing  # this should be done after initQgis and will use the custom pathfinder
    from processing.core.Processing import Processing  # this is needed so the algorithms are registered

    Processing.initialize()

    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))

    logging.info("Init Qgis done")

    # Get layer with name EL.GridCoverage from WMS 'https://www.geoseaportal.de/geoserver/ELC_INSPIRE/ows?VERSION=1.3.0'
    layer = QgsRasterLayer(
        'contextualWMSLegend=0&crs=EPSG:4326&dpiMode=7&featureCount=10&format=image/png&layers=EL.GridCoverage&styles&tilePixelRatio=0&url=https://www.geoseaportal.de/geoserver/ELC_INSPIRE/ows?VERSION%3D1.3.0',
        'EL.GridCoverage', 'wms')
    if not layer.isValid():
        raise Exception("Layer failed to load!")
    layer.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    project.addMapLayer(layer)
    logging.info("Added EL.GridCoverage")

    # get bands
    bands = layer.dataProvider().bandCount()
    logging.info(f"Layer has {bands} bands")

    return qgs, layer


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    logging.warning(f"""

The script needs to be run in the PyQuis Python Environment and was tested on QGIS 3.34.1 (non LTR).
You might need to manually adjust the QGISPATH variable in the script.  
""")
    from Classes import GpsPoint

    logging.info("Starting")

    qgs, layer = create_layers()
    raw_data_path = "data/raw/"
    with open(raw_data_path + "raw_data.json", "r") as f:
        raw_data = json.load(f)

    data_filtered = list(filter(lambda data: len(data["ship_info"].values()) > 0, raw_data))

    for data in data_filtered:
        logging.info(f"Processing {data['station_info']['station_name']}")
        google_maps_link = data["basic_info"]["location"]
        coords = GpsPoint.from_google_maps_link(google_maps_link)
        point = QgsPointXY(coords.longitude, coords.latitude)
        z = get_depth_at_point(layer, point)
        data["station_info"]["calculated_depth"] = z
        logging.debug(f"Station {data['station_info']['station_name']} has depth {z}")

    logging.info("Writing data to file")
    with open(raw_data_path + "raw_data_with_depth.json", "w") as f:
        json.dump(raw_data, f, indent=4)
    logging.info("Done")
