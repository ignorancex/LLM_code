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

import random
import sys
import zipfile
import os
import logging
import importlib
import importlib.util
import datetime
from importlib.abc import MetaPathFinder
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
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


sys.meta_path.insert(0, ProcessingPathFinder)

# now we can finally import qgis
from qgis.core import *
from qgis.PyQt.QtCore import QVariant


def get_borders():
    if not os.path.exists(DATAFOLDER):
        os.makedirs(DATAFOLDER)

    # check if file DATAFOLDER/borders exists and if not download it
    if not os.path.exists(f"{DATAFOLDER}/borders.gpkg"):
        if not os.path.exists(f"{DATAFOLDER}/borders.gpkg.zip"):
            logging.info("Downloading borders")
            import urllib.request
            # i would prefer to use requests but it is not included in the PyQuis Python Environment
            # and installing additional packages to it should be avoided
            url = "https://daten.gdz.bkg.bund.de/produkte/sonstige/geogitter/2019/DE_Grid_ETRS89-LAEA_100m.gpkg.zip"

            def report(blocknr, blocksize, size):
                current = blocknr * blocksize  # doing this in logging is just to much
                sys.stdout.write("\rDownload Progress {0:.2f}%".format(100.0 * current / size))
                sys.stdout.flush()

            urllib.request.urlretrieve(url, f"{DATAFOLDER}/borders.gpkg.zip", reporthook=report)
            logging.info("Downloading borders done")

        logging.info("Starting to validate hash")
        # validate hash of zip file+
        hash = hashlib.md5()
        with open(f"{DATAFOLDER}/borders.gpkg.zip", "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        hash = hash.hexdigest()
        logging.debug("Hash of borders zip file: " + hash)
        assert hash == "0ec44d6e62f43a55208afb348b30fc79", "Hash of borders zip file is not correct"

        logging.info("Starting to unzip")

        # unzip file
        with zipfile.ZipFile(f"{DATAFOLDER}/borders.gpkg.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{DATAFOLDER}/unzip")
        # search for gpkg file recursivly in unzip folder and copy it to DATAFOLDER
        import shutil
        for root, dirs, files in os.walk(f"{DATAFOLDER}/unzip"):
            for file in files:
                if file.endswith(".gpkg"):
                    shutil.copy(os.path.join(root, file), f"{DATAFOLDER}/borders.gpkg")
                if file.endswith(".pdf") or file.endswith(".txt"):
                    # copy metadata files such as license information
                    shutil.copy(os.path.join(root, file), DATAFOLDER)
        # delete unzip folder
        shutil.rmtree(f"{DATAFOLDER}/unzip")

        logging.info("Unzipping done")

    # calculate hash of borders file
    hash = hashlib.md5()
    with open(f"{DATAFOLDER}/borders.gpkg", "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    hash = hash.hexdigest()

    logging.debug("Hash of borders file: " + hash)
    assert hash == "70ffd94179c26b76fe15b50f6294c2c9", "Hash of borders file is not correct"


def create_n_points_in_polygon(polygon, n, seed, origin_crs, target_crs=4326):
    random.seed(seed)
    geom = polygon.geometry()
    engine = QgsGeometry.createGeometryEngine(geom.constGet())
    engine.prepareGeometry()

    bbox = geom.boundingBox()

    points = []
    while len(points) < n:
        x = random.uniform(bbox.xMinimum(), bbox.xMaximum())
        y = random.uniform(bbox.yMinimum(), bbox.yMaximum())
        point = QgsPointXY(x, y)
        if engine.contains(QgsGeometry.fromPointXY(point).constGet()):
            point = QgsPoint(point)
            point.transform(QgsCoordinateTransform(origin_crs, QgsCoordinateReferenceSystem.fromEpsgId(target_crs),
                                                   QgsProject.instance()))
            points.append(point)
    return points


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


def generate_point_files(foldername, feature, border_layer):
    folder = os.path.join(DATAFOLDER, foldername)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for count in (10, 100, 500, 1_000, 5_000, 10_000):
        for seed in range(1, 1 + 10):
            points = create_n_points_in_polygon(feature, count, seed, border_layer.crs())
            with open(os.path.join(folder, str(count) + "_" + str(seed) + ".csv"), "w") as f:
                f.write("lat,lon\n")
                for point in points[:-1]:
                    f.write(str(point.y()) + "," + str(point.x()) + "\n")
                f.write(str(points[-1].y()) + "," + str(points[-1].x()))  # no newline at end of file
            logging.info(f"Created {count} points for {foldername} with seed {seed}")


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

    logging.info("Init Qgis done")

    # create vector layer from borders file
    border_layer = QgsVectorLayer(f"{DATAFOLDER}/borders.gpkg", "borders", "ogr")
    if not border_layer.isValid():
        raise Exception("Layer failed to load!")
    else:
        project.addMapLayer(border_layer)
        logging.info("Layer was loaded successfully!")
    border_layer.updateExtents()

    # filter by p_wasser=100
    logging.info("Filtering layer")
    water_layer = border_layer.materialize(QgsFeatureRequest().setFilterExpression("p_wasser=100"))

    # dissolve layer
    logging.info("Dissolving layer")
    feedback = LoggingFeedback()
    res = processing.run("native:dissolve", {
        'FIELD': [],
        'INPUT': water_layer,
        'OUTPUT': 'TEMPORARY_OUTPUT',
        'SEPARATE_DISJOINT': True
    }, feedback=feedback)
    if res['OUTPUT'] is None:
        raise Exception("Dissolving layer failed")
    else:
        border_layer = res['OUTPUT']
        project.addMapLayer(border_layer)
        logging.info("Dissolving layer done")

    # order by area
    logging.info("Ordering layer")
    # set area to geometry area
    border_layer.startEditing()
    border_layer.addAttribute(QgsField("area", QVariant.Double))
    for feature in border_layer.getFeatures():
        feature["area"] = feature.geometry().area()
        border_layer.updateFeature(feature)
    border_layer.commitChanges()
    request = QgsFeatureRequest().addOrderBy("area", False)
    request.setLimit(2)
    border_layer = border_layer.materialize(request)

    project.addMapLayer(border_layer)
    northern_sea = border_layer.getFeature(1)
    eastern_sea = border_layer.getFeature(2)

    # combine northern and eastern sea
    logging.info("Combining features")
    feedback = LoggingFeedback()
    res = processing.run("native:collect", {
        'INPUT': border_layer,
        'FIELD': [], 'OUTPUT': 'TEMPORARY_OUTPUT'},
                         feedback=feedback)

    if res['OUTPUT'] is None:
        raise Exception("Combining features failed")
    combined_layer = res['OUTPUT']
    combined = next(combined_layer.getFeatures())
    border_layer.startEditing()
    border_layer.addFeature(combined)
    border_layer.commitChanges()
    logging.info("Combining features done")

    return qgs, northern_sea, eastern_sea, combined, border_layer


if __name__ == '__main__':
    print(f"""
=======================
DGZRS QGIS Zone Creator    
=======================
This script will create zone data for the DGzRS project.
It will create a folder data/geo and download the borders file from 
https://gdz.bkg.bund.de/index.php/default/open-data/geographische-gitter-fur-deutschland-in-utm-projektion-geogitter-national.html
Copyright of the downloaded data (not the script / the produced data):
© GeoBasis-DE / BKG {datetime.datetime.now().year}
The data is licensed under the Data licence Germany – attribution – version 2.0

The script will then create points in the northern sea and eastern sea.

The script needs to be run in the PyQuis Python Environment and was tested on QGIS 3.34.1 (non LTR).
You might need to manually adjust the QGISPATH variable in the script.  
""")

    logging.info("Starting")
    logging.info("Getting borders")
    get_borders()
    logging.info("Getting borders done")

    qgs, northern_sea, eastern_sea, combined, layer = create_layers()

    # create points in northern sea
    logging.info("Creating points in northern sea")
    generate_point_files("northern_sea", northern_sea, layer)
    logging.info("Creating points in northern sea done")

    # create points in eastern sea
    logging.info("Creating points in eastern sea")
    generate_point_files("eastern_sea", eastern_sea, layer)
    logging.info("Creating points in eastern sea done")

    # create points in northern + eastern sea
    logging.info("Creating points in northern + eastern sea")
    generate_point_files("combined", combined, layer)
    logging.info("Creating points in northern + eastern sea done")

    # When your script is complete, call exitQgis() to remove the provider and
    # layer registries from memory
    qgs.exitQgis()
    logging.info("Done")
