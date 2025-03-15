import requests
import csv
from io import StringIO
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import dict_uris
from xnat2mids.xnat.request import try_to_request
from xnat2mids.xnat.scan_resources import ScanResources


class Scan(dict):
    def __init__(self, session, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["session"] = session
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_resources(self, verbose):
        output = StringIO()
        if verbose: print(format_message(self.level_verbose, self.level_tab, f"scan: {self['ID']}"))
        output.write(
            try_to_request(
                self["session"]["subject"]["project"].interface,
                self["session"]["subject"]["project"].url_xnat
                + dict_uris["scan_resources"](
                    self["session"]["subject"]["project"]["ID"],
                    self["session"]["subject"]["ID"],
                    self["session"]["ID"],
                    self["ID"]
                )
            ).text
        )
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_resources = dict()
        for row in reader:
            self.dict_resources[row["xnat_abstractresource_id"]] = ScanResources(self, self.level_verbose + 1, self.level_tab + 1, **row)
        output.close()

    def download(
            self, path_download,
            bool_list_resources=[False, False, True, False, False, False],
            overwrite=False,
            verbose=False
    ):
        #print(format_message(13, 4, "entro en scan"))
        self.get_list_resources(verbose)
        for resource_obj in self.dict_resources.values():
            try:
                resource_obj.download(
                    path_download,
                    bool_list_resources=bool_list_resources,
                    overwrite=overwrite, verbose=verbose)
            except requests.exceptions.RequestException as e:
                print(e)
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)
