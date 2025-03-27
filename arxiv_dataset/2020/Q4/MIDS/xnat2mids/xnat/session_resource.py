import csv
import getpass
import json
import os
import re
import sys
import time
import traceback
from io import StringIO
from shutil import copyfile
import requests
from xnat2mids.xnat.request import try_to_request
from xnat2mids.xnat.variables import types_files_xnat
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import reset_terminal
from xnat2mids.xnat.variables import dict_paths
from xnat2mids.xnat.variables import dict_uris




class SessionResource(dict):
    def __init__(self, session, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["session"] = session
        self.level_verbose = level_verbose
        self.level_tab = level_tab


    def get_list_files(self, verbose):
        output = StringIO()
        if verbose:
            print(format_message(
                self.level_verbose, self.level_tab, f"Session resource file: {self['label']}"),
                end=" ----> ",
                flush=True
            )
        u = self["session"]["subject"]["project"].url_xnat + dict_uris["session_resource_files"](
                self["session"]["subject"]["project"]["ID"],
                self["session"]["subject"]["ID"],
                self["session"]["ID"],
                self["label"]
            )
        output.write(
            try_to_request(
            self["session"]["subject"]["project"].interface,
            self["session"]["subject"]["project"].url_xnat
            + dict_uris["session_resource_files"](
                    self["session"]["subject"]["project"]["ID"],
                    self["session"]["subject"]["ID"],
                    self["session"]["ID"],
                    self["label"]
                )
            ).text
        )


        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_resources = dict()
        for row in reader:
            self.dict_resources[row["Name"]] = dict(**row)
        output.close()


    def download_resource_file(self, path_download, filename, overwrite=False, verbose=False):
        complet_path = (path_download.joinpath(dict_paths["path_resources"](
            self["session"]["subject"]["ID"],
            self["session"]["ID"],
            self["label"]
        ))
                        )

        resource_path = os.path.join(complet_path, filename)
        if not overwrite and os.path.exists(resource_path):
            if verbose: print("resource file already exist")
            return
        if verbose: print("Downloading resource file...", flush=True)
        os.makedirs(complet_path, exist_ok=True)
        url_resource = (self["session"]["subject"]["project"].url_xnat
                        + dict_uris["session_resource_files"](
                    self["session"]["subject"]["project"]["ID"],
                    self["session"]["subject"]["ID"],
                    self["session"]["ID"],
                    self["label"]
                ).split("?")[0] + "/"
                        + filename
                        )
        resource = try_to_request(
            self["session"]["subject"]["project"].interface,
            url_resource
        )
        # png = self["scan"]["session"]["subject"]["project"].interface.get(url_png, allow_redirects=True)
        resource.raise_for_status()

        with open(os.path.join(complet_path, filename), 'wb') as png_file:
            png_file.write(resource.content)

    def download(
            self,
            path_download,
            bool_list_resources=[False, False, True, False, False, False],
            overwrite=False,
            verbose=False
    ):

        self.get_list_files(verbose)
        is_metadata_saved = False

        for index, file_obj in enumerate(self.dict_resources.values()):
            self.download_resource_file(
                path_download, file_obj["Name"], overwrite=overwrite, verbose=verbose
            )

        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)
        print(format_message(self.level_verbose + 1, self.level_tab + 1, "\u001b[0K"), end="", flush=True)


