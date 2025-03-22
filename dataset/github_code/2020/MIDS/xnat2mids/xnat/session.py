
import csv
from io import StringIO
from xnat2mids.xnat.scan import Scan
from xnat2mids.xnat.session_resource import SessionResource
from xnat2mids.xnat.assessor import Assessors
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import dict_uris
from xnat2mids.xnat.request import try_to_request


class Session(dict):
    def __init__(self, subject, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["subject"] = subject
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_scans(self, verbose):
        output = StringIO()
        if verbose: print(
            format_message(self.level_verbose, self.level_tab, f"Session: {self['label']}"), flush=True)
        output.write(
            try_to_request(
                self["subject"]["project"].interface,
                self["subject"]["project"].url_xnat
                + dict_uris["scans"](
                    self["subject"]["project"]["ID"],
                    self["subject"]["ID"],
                    self["ID"])
            ).text
        )
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_scans = dict()
        for row in reader:
            #print(format_message(16, 5, ""))
            #print(row)
            # If the "xsiType" key of the session is not equal to
            # "xnat: mrSessionData", this may not work in this point.
            try:
                self.dict_scans[row["ID"]] = Scan(self, self.level_verbose + 1, self.level_tab + 1, **row)

            except KeyError:
                continue
        output.close()
    def get_list_session_resources(self, path_download, overwrite=False, verbose=False):
        output = StringIO()
        if verbose: print(
            format_message(self.level_verbose, self.level_tab, f"Session: {self['ID']}"), flush=True)
        u = (
            self["subject"]["project"].url_xnat
            + dict_uris["session_resources"](
                self["subject"]["project"]["ID"],
                self["subject"]["ID"],
                self["ID"]
            )
        )
        output.write(
            try_to_request(
                self["subject"]["project"].interface,
                self["subject"]["project"].url_xnat
                + dict_uris["session_resources"](
                    self["subject"]["project"]["ID"],
                    self["subject"]["ID"],
                    self["ID"])
            ).text
        )
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_resources = dict()
        for row in reader:

            try:
                self.dict_resources[row["label"]] = SessionResource(self, self.level_verbose + 1, self.level_tab + 1, **row)

            except KeyError:
                continue
        output.close()
    # def get_list_struct_report(
    #         self, path_download, bool_list_resources, overwrite=False, verbose=False
    # ):
    #     output = StringIO()
    #     output.write(
    #         try_to_request(
    #             self["subject"]["project"].interface,
    #             self["subject"]["project"].url_xnat
    #             + dict_uris["struct_reports"](
    #                 self["subject"]["project"]["ID"],
    #                 self["subject"]["ID"],
    #                 self["ID"])
    #         ).text
    #     )
    #     output.seek(0)
    #     reader = csv.DictReader(output)
    #
    #     if verbose:
    #         print(format_message(self.level_verbose, self.level_tab, f"Session SR:  {self['ID']}"), flush=True)
    #     for row in reader:
    #         # id_session=re.search(r'\d+', row["Name"]).group(0)
    #         try:
    #             id_session = re.search(r'sr.*', row["Name"]).group(0)
    #         except KeyError as e:
    #             print("\033[20;0H\u001b[0K", end="", flush=True)
    #             print(row)
    #             print(self["subject"]["project"]["ID"],
    #                   self["subject"]["ID"],
    #                   self["ID"])
    #             return
    #         self.download_struct_report(
    #             path_download, row["Name"], overwrite=overwrite, verbose=verbose
    #         )
    #     output.close()
    #
    # def download_struct_report(self, path_download, filename, overwrite=False, verbose=False):
    #
    #     complet_path = (
    #         os.path.join(
    #             path_download,
    #             dict_path["path_sr"](
    #                 self["subject"]["ID"],
    #                 self["ID"]
    #             )
    #         )
    #     )
    #     sr_path = os.path.join(complet_path, filename)
    #     if not overwrite and os.path.exists(sr_path):
    #         print("Structural Report exists", flush=True)
    #         return
    #     os.makedirs(complet_path, exist_ok=True)
    #     url_sr = (self["subject"]["project"].url_xnat
    #               + dict_uri["struct_reports"](
    #                 self["subject"]["project"]["ID"],
    #                 self["subject"]["ID"],
    #                 self["ID"]
    #             ).split("?")[0] + os.sep
    #               + filename
    #               )
    #     # if verbose: print("          Downloading sr file... " + url_sr,flush=True)
    #     sr = try_to_request(
    #         self["subject"]["project"].interface,
    #         url_sr
    #     )
    #     # nifti = self["scan"]["session"]["subject"]["project"].interface.get(url_nifti, allow_redirects=True)
    #     sr.raise_for_status()
    #
    #     with open(os.path.join(complet_path, filename), 'wb') as sr_file:
    #         sr_file.write(sr.content)

    def get_list_assessors(self, verbose):
        output = StringIO()
        if verbose:
            print(format_message(self.level_verbose, self.level_tab, f"Assessors:  {self['ID']}"), flush=True)
        output.write(
            try_to_request(
                self["subject"]["project"].interface,
                self["subject"]["project"].url_xnat
                + dict_uris["assessors"](
                    self["subject"]["project"]["ID"],
                    self["subject"]["ID"],
                    self["ID"]

                )
            ).text
        )
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_assessors = dict()
        for row in reader:
            try:
                self.dict_assessors[row["ID"]] = Assessors(self, self.level_verbose + 1, self.level_tab + 1, **row)
            except Exception as e:
                print(e)
                # sys.exit(2)
                continue
        output.close()

    def download(self, path_download,
                 bool_list_resources=[False, False, True, False, False, False],
                 overwrite=False,
                 verbose=False
                 ):
        # print("\033[7;0H\u001b[0K", end="",flush=True)
        self.get_list_scans(verbose)
        for scan_obj in self.dict_scans.values():
            # if "anat" in scan_obj["ID"]:
                scan_obj.download(
                    path_download, bool_list_resources=bool_list_resources,
                    overwrite=overwrite, verbose=verbose
                )

        # print("\033[18;0H\u001b[0K", end="",flush=True)
        #self.get_list_struct_report(path_download, bool_list_resources, overwrite=False, verbose=verbose)

        # self.get_list_assessors(verbose)
        # for resource_obj in self.dict_assessors.values():

        #     resource_obj.download(
        #         path_download,
        #         bool_list_resources=bool_list_resources,
        #         overwrite=overwrite, verbose=verbose)

        self.get_list_session_resources(verbose)
        for resource_obj in self.dict_resources.values():
                resource_obj.download(
                    path_download,
                    bool_list_resources=bool_list_resources,
                    overwrite=overwrite, verbose=verbose)
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)

