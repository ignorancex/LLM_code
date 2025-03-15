

import csv
import os
from pathlib import Path
from io import StringIO
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import dict_uris
from xnat2mids.xnat.variables import dict_paths
from xnat2mids.xnat.request import try_to_request
from xnat2mids.xnat.convert_xml2image import xml2image

class AssessorsResources(dict):
    def __init__(self, assessors, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["assessors"] = assessors
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_roi_files(self, verbose):
        output = StringIO()
        if verbose:
            print(
                format_message(
                    self.level_verbose, self.level_tab, f"assessor resources: {self['label']}"
                ), end=" ----> ", flush=True
            )
        file_text = try_to_request(
            self["assessors"]["session"]["subject"]["project"].interface,
            self["assessors"]["session"]["subject"]["project"].url_xnat
            + dict_uris["assessor_resource_roi_files"](
                self["assessors"]["session"]["subject"]["project"]["ID"],
                self["assessors"]["session"]["subject"]["ID"],
                self["assessors"]["session"]["ID"],
                self["assessors"]["ID"],
                self["xnat_abstractresource_id"]
            )
        )
        file_text.raise_for_status()
        output.write(file_text.text)
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_roi_files = dict()
        for row in reader:
            self.dict_roi_files[row["Name"]] = dict(**row)
        output.close()

    def download_roi_files(self, path_download, filename, overwrite=False, verbose=False):
        complet_path = (path_download + dict_paths["path_download_roi"](
            self["assessors"]["session"]["subject"]["ID"],
            self["assessors"]["session"]["ID"],
            self["assessors"]["ID"],
            self["xnat_abstractresource_id"]
        )
                        )

        roi_path = os.path.join(complet_path, filename)
        if not overwrite and os.path.exists(roi_path):
            if verbose: print(" roi file already exist")
            xml2image(Path(roi_path))
            return
        if verbose: print(" Downloading png file...", flush=True)
        os.makedirs(complet_path, exist_ok=True)
        url_roi = (self["assessors"]["session"]["subject"]["project"].url_xnat
                   + dict_uris["assessor_resource_roi_files"](
                    self["assessors"]["session"]["subject"]["project"]["ID"],
                    self["assessors"]["session"]["subject"]["ID"],
                    self["assessors"]["session"]["ID"],
                    self["assessors"]["ID"],
                    self["xnat_abstractresource_id"]
                ).split("?")[0] + "/"
                   + filename
                   )

        roi = try_to_request(
            self["assessors"]["session"]["subject"]["project"].interface,
            url_roi
        )

        # png = self["scan"]["session"]["subject"]["project"].interface.get(url_png, allow_redirects=True)
        roi.raise_for_status()

        with open(roi_path, 'wb') as roi_file:
            roi_file.write(roi.content)
        #xml2image(Path(roi_path))

    def download(
            self,
            path_download,
            bool_list_resources=[False, False, True, False, False, False],
            overwrite=False,
            verbose=False
    ):
        self.get_list_roi_files(verbose)
        for file_obj in self.dict_roi_files.values():
            self.download_roi_files(path_download, file_obj["Name"], overwrite=overwrite, verbose=verbose)
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)