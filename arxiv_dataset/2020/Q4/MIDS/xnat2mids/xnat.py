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

import progressbar
import pydicom
from xnat import requests
from xnat.xnat.variables import dict_paths
from xnat.xnat.variables import dict_uris
from xnat.xnat.variables import format_message
from xnat.xnat.variables import reset_terminal


###############################################################################
# Functions
###############################################################################
def list_directory_xnat(project_list):
    project_list.sort()
    loop = True
    project_names = [path_.split(os.sep)[-1] for path_ in project_list]
    while loop:
        for i, project_item in enumerate(project_names):
            if not i % 5:
                print((""))
            string = str(i + 1) + ") " + project_item
            print("{0:20s}".format(string), end=" ", flush=True)
        answer = input("\nChoose the project: ")
        answer = str(answer)
        answer_list = []
        answer_list_aux = answer.split(' ')

        for ans in answer_list_aux:
            if ans.isdigit():
                ans = int(ans) - 1
                if ans >= len(project_names):
                    print("the number " + ans + " is not corrected, try again", flush=True)
                    break
                else:
                    answer_list.append(project_names[ans])
            else:
                if not (any([ans in path_ for path_ in project_names])):
                    if ans == "exit":
                        exit(1)
                    else:
                        print("the project " + ans
                              + " is not corrected, try again")
                        break
                else:
                    answer_list.append(ans)
        else:
            loop = False

    return answer_list


def try_to_request(interface, web, level_verbose=15, level_tab=15):
    # Function that allows a persistent get request to a web page

    # variable that counts connection attempt
    intents = 1
    while True:
        try:
            req = interface.get(web)
            if req.status_code != 504:
                break
        except requests.exceptions.ConnectionError as e:

            print(
                format_message(
                    level_verbose,
                    level_tab,
                    "Intents to download information = {}".format(intents)
                )
            )
            time.sleep(60)
        except requests.exceptions.HTTPError:
            print(
                format_message(
                    level_verbose,
                    level_tab,
                    "Intents to download information = {}".format(intents)
                )
            )
            time.sleep(60)
    return req


class Xnat_Session():

    def __init__(self, url_xnat, user=None, password=None):
        self.url_xnat = url_xnat
        self.user = user
        self.password = password
        self.level_verbose = 0
        self.level_tab = 0

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def connect(self):
        self.interface = requests.Session()
        if self.user and self.password:
            self.interface.auth = (self.user, self.password)
        else:
            if not self.user:
                self.user = input("username: ")
            self.password = getpass.getpass("Password: ")
            self.interface.auth = (self.user, self.password)

    def exit(self):
        self.interface.close()

    def get_projects(self, verbose):
        output = StringIO()
        if verbose: print(self.url_xnat + dict_uris["projects"], flush=True)
        output.write(
            try_to_request(
                self.interface, self.url_xnat
                                + dict_uris["projects"],
                self.level_verbose,
                self.level_tab
            ).text
        )
        output.seek(0)
        print(str(output))
        reader = csv.DictReader(output)
        self.dict_projects = dict()
        for row in reader:
            self.dict_projects[row["ID"]] = Project(self.url_xnat, self.interface, self.level_verbose + 5,
                                                    self.level_tab + 5, **row)
        output.close()

    def show_list_of_project(self, verbose):
        """
        This functions allows the user to visualize al projects in xnat aplication
        NOT IN USE, NEXT UPLOAD
        """
        project_list = [key for key, _ in self.dict_projects.items()]
        answer_list = list_directory_xnat(project_list)

        # project_list.sort()
        # loop = True
        # while loop:
        #     for i, project_item in enumerate(project_list):
        #         if not i % 5:
        #             print((""))
        #         string = str(i + 1) + ") " + project_item
        #         print("{0:20s}".format(string), end=" ", flush=True)
        #     answer = input("\nChoose the project: ")
        #     answer = str(answer)
        #     answer_list = []
        #     answer_list_aux = answer.split(' ')
        #     for ans in answer_list_aux:
        #         if ans.isdigit():
        #             ans = int(ans) - 1
        #             if ans >= len(project_list):
        #                 print("the number " + ans + " is not corrected, try again", flush=True)
        #                 break
        #             else:
        #                 answer_list.append(project_list[ans])
        #         else:
        #             if not (ans in project_list):
        #                 if ans == "exit":
        #                     exit(1)
        #                 else:
        #                     print("the proyect " + ans
        #                           + " is not corrected, try again")
        #                     break
        #             else:
        #                 answer_list.append(ans)
        #     else:
        #         loop = False
        if verbose: print("list of projects to download: {}".format(", ".join(answer_list)), flush=True)
        return answer_list

    def download_projects(self,
                          path_download,
                          with_department=True,
                          project_list=None,
                          bool_list_resources=[False, False, True, False, False, False],
                          overwrite=False,
                          verbose=False):
        self.get_projects(verbose)
        if project_list is None:
            project_list = self.show_list_of_project(verbose)
        # clear console
        print(reset_terminal, end="", flush=True)
        print(format_message(self.level_verbose, self.level_tab, "Projects:"), end="", flush=True)
        print(format_message(self.level_verbose + 3, self.level_tab, "Subject:"), end="", flush=True)
        # move the cursor
        bar_project = progressbar.ProgressBar(maxval=len(project_list), prefix="\033[2;0H").start()
        bar_project.update(1)
        bar_project.update(0)

        for iter, key in enumerate(project_list):
            self.dict_projects[key].download(
                os.path.join(path_download, ""),
                with_department=with_department,
                bool_list_resources=bool_list_resources,
                overwrite=overwrite,
                verbose=verbose
            )
            bar_project.update(iter + 1)
        bar_project.finish()
        print("\033[12:0", end="", flush=True)


class Project(dict):
    def __init__(self, url_xnat, interface, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self.url_xnat = url_xnat
        self.interface = interface
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_subjects(self, verbose):
        output = StringIO()

        if verbose:
            print(format_message(self.level_verbose, self.level_tab, "Project: {}".format(self["ID"])), flush=True)
        output.write(try_to_request(self.interface, self.url_xnat
                                    + dict_uris["subjects"](self["ID"])
                                    ).text)
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_subjects = dict()
        for row in reader:
            row["project"] = self
            self.dict_subjects[row["ID"]] = Subject(self.level_verbose + 1, self.level_tab + 1, **row)
        output.close()

    def download(
            self,
            path_download,
            with_department=True,
            project_list=None,
            bool_list_resources=[False, False, True, False, False, False],
            overwrite=False,
            verbose=False
    ):
        if with_department: path_download = os.path.join(path_download, self["ID"], "")
        # print("\033[5;0H\u001b[0K", end="",flush=True)
        self.get_list_subjects(verbose)
        # move the cursor
        # print("\033[4;0H", end="",flush=True)
        bar_subject = progressbar.ProgressBar(
            maxval=len(self.dict_subjects),
            prefix=format_message(self.level_verbose - 1, 0, "")
        ).start()
        for iter, subject_obj in enumerate(self.dict_subjects.values()):
            subject_obj.download(
                path_download, bool_list_resources,
                overwrite=overwrite, verbose=verbose
            )
            print(format_message(self.level_verbose - 1, 0, ""))
            bar_subject.update(iter + 1)
        print(format_message(self.level_verbose - 1, 0, ""))
        bar_subject.finish()


class Subject(dict):
    def __init__(self, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_experiments(self, verbose):

        output = StringIO()
        if verbose: print(format_message(self.level_verbose, self.level_tab, f"Subject: {self['ID']}"), flush=True)
        output.write(
            try_to_request(
                self["project"].interface,
                self["project"].url_xnat
                + dict_uris["experiments"](self["project"]["ID"], self["ID"])
            ).text
        )
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_sessions = dict()
        for row in reader:
            self.dict_sessions[row["label"]] = Session(self, self.level_verbose + 1, self.level_tab + 1, **row)
        output.close()

    def download(self, path_download,
                 bool_list_resources=[False, False, True, False, False, False],
                 overwrite=False,
                 verbose=False
                 ):
        print("\033[6;0H\u001b[0K", end="", flush=True)
        self.get_list_experiments(verbose)
        # move the cursor
        for session_obj in self.dict_sessions.values():
            session_obj.download(
                path_download, bool_list_resources,
                overwrite=overwrite, verbose=verbose
            )

        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)


class Resource(dict):
    pass


class Session(dict):
    def __init__(self, subject, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["subject"] = subject
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_scans(self, verbose):
        output = StringIO()
        if verbose: print(
            format_message(self.level_verbose, self.level_tab, f"Session: {self['ID']}"), flush=True)
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
            # print(format_message(16, 5, ""))
            # print(row)
            # If the "xsiType" key of the session is not equal to
            # "xnat: mrSessionData", this may not work in this point.
            try:
                self.dict_scans[row["ID"]] = scan(self, self.level_verbose + 1, self.level_tab + 1, **row)

            except KeyError:
                continue
        output.close()

    def get_list_resources(self, path_download, overwrite=False, verbose=False):
        output = StringIO()
        if verbose: print(
            format_message(self.level_verbose, self.level_tab, f"Session: {self['ID']}"), flush=True)
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
            # print(format_message(16, 5, ""))
            # print(row)
            # If the "xsiType" key of the session is not equal to
            # "xnat: mrSessionData", this may not work in this point.
            try:
                self.dict_scans[row["ID"]] = Resource(self, self.level_verbose + 1, self.level_tab + 1, **row)

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
                self.dict_assessors[row["ID"]] = assessors(self, self.level_verbose + 1, self.level_tab + 1, **row)
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
            scan_obj.download(
                path_download, bool_list_resources=bool_list_resources,
                overwrite=overwrite, verbose=verbose
            )
        # print("\033[18;0H\u001b[0K", end="",flush=True)
        self.get_list_struct_report(path_download, bool_list_resources, overwrite=False, verbose=verbose)

        self.get_list_assessors(verbose)
        for resource_obj in self.dict_assessors.values():
            # sys.exit(1)
            resource_obj.download(
                path_download,
                bool_list_resources=bool_list_resources,
                overwrite=overwrite, verbose=verbose)
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)


class scan(dict):
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
            self.dict_resources[row["xnat_abstractresource_id"]] = resources(self, self.level_verbose + 1,
                                                                             self.level_tab + 1, **row)
        output.close()

    def download(
            self, path_download,
            bool_list_resources=[False, False, True, False, False, False],
            overwrite=False,
            verbose=False
    ):
        # print(format_message(13, 4, "entro en scan"))
        self.get_list_resources(verbose)
        for resource_obj in self.dict_resources.values():
            try:
                resource_obj.download(
                    path_download,
                    bool_list_resources=bool_list_resources,
                    overwrite=overwrite, verbose=verbose)
            except requests.exceptions.RequestException as e:
                complet_path = (path_download + dict_path["path_download"](
                    self["session"]["subject"]["ID"],
                    self["session"]["ID"],
                    self["ID"],
                    "NIFTI"
                )
                                )
                print(e)
                # with open(os.path.join(path_download, ".log"), "a") as log:
                #    log.write(
                #        "Error: {} in nifti url {}".format(
                #            e,
                #            complet_path
                #        )

                #    )
                # return
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)


class scan_resources(dict):
    def __init__(self, scan, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["scan"] = scan
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_files(self, verbose):
        output = StringIO()
        if verbose: print(format_message(self.level_verbose, self.level_tab, f"resource: {self['label']}"),
                          end=" ----> ",
                          flush=True)
        file_text = try_to_request(
            self["scan"]["session"]["subject"]["project"].interface,
            self["scan"]["session"]["subject"]["project"].url_xnat
            + dict_uri["files"](
                self["scan"]["session"]["subject"]["project"]["ID"],
                self["scan"]["session"]["subject"]["ID"],
                self["scan"]["session"]["ID"],
                self["scan"]["ID"],
                self["label"]
            )
        )
        file_text.raise_for_status()
        output.write(file_text.text)
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_resources = dict()
        for row in reader:
            self.dict_resources[row["Name"]] = dict(**row)
        output.close()

    def download_dicom(self, path_download, filename, overwrite=False, verbose=False):

        complet_path = (
                path_download + dict_path["path_download"](
            self["scan"]["session"]["subject"]["ID"],
            self["scan"]["session"]["ID"],
            self["scan"]["ID"],
            self["label"]
        )
        )
        complet_path_metadata = (
                path_download + dict_path["path_download"](
            self["scan"]["session"]["subject"]["ID"],
            self["scan"]["session"]["ID"],
            self["scan"]["ID"],
            "NIFTI"
        )
        )

        dicom_path = os.path.join(complet_path, filename)
        dicom_path_metadata = os.path.join(complet_path_metadata, "dicom.json")
        if not overwrite and (os.path.exists(dicom_path) and os.path.exists(dicom_path_metadata)):
            # if verbose: print("DICOM file already exist")
            return dicom_path, dicom_path_metadata
        # if not overwrite and os.path.exists(dicom_path):
        #    if verbose: print("          DICOM metadata file already exist")
        #    return
        if verbose:
            pass
            # print("Downloading DICOM file...", flush=True, end="")
            # print("          Downloading DICOM  Metadata file...",flush=True)

        os.makedirs(complet_path, exist_ok=True)
        os.makedirs(complet_path_metadata, exist_ok=True)
        try:
            copyfile(dict_uri["local_files"](
                self["scan"]["session"]["subject"]["project"]["ID"],
                self["scan"]["session"]["label"],
                self["scan"]["ID"],
                self["label"],
                filename

            ), dicom_path)
        except FileNotFoundError as e:
            url_dicom = (self["scan"]["session"]["subject"]["project"].url_xnat
                         + dict_uri["files"](
                        self["scan"]["session"]["subject"]["project"]["ID"],
                        self["scan"]["session"]["subject"]["ID"],
                        self["scan"]["session"]["ID"],
                        self["scan"]["ID"],
                        self["label"]
                    ).split("?")[0] + "/"
                         + filename
                         )

            # with open(path_download + "urls_descarga.txt", "a") as f:
            #    f.write("%s \n %s # %r # %r # %r \n" % (
            #        url_dicom, dicom_path, overwrite, verbose, os.path.exists(dicom_path)))
            #    f.close()

            dicom = try_to_request(
                self["scan"]["session"]["subject"]["project"].interface,
                url_dicom
            )
            # nifti = self["scan"]["session"]["subject"]["project"].interface.get(url_nifti, allow_redirects=True)
            dicom.raise_for_status()

            with open(dicom_path, 'wb') as dicom_file:
                dicom_file.write(dicom.content)

        return dicom_path, dicom_path_metadata

    def download_nifti(self, path_download, filename, overwrite=False, verbose=False):
        complet_path = (path_download + dict_path["path_download"](
            self["scan"]["session"]["subject"]["ID"],
            self["scan"]["session"]["ID"],
            self["scan"]["ID"],
            self["label"]
        )
                        )

        nifti_path = os.path.join(complet_path, filename)
        if not overwrite and os.path.exists(nifti_path):
            if verbose: print("nifti file already exist")
            return
        if verbose: print("Downloading NIFTI file...", flush=True)
        os.makedirs(complet_path, exist_ok=True)
        url_nifti = (self["scan"]["session"]["subject"]["project"].url_xnat
                     + dict_uri["files"](
                    self["scan"]["session"]["subject"]["project"]["ID"],
                    self["scan"]["session"]["subject"]["ID"],
                    self["scan"]["session"]["ID"],
                    self["scan"]["ID"],
                    self["label"]
                ).split("?")[0] + "/"
                     + filename
                     )

        # with open(path_download + "urls_descarga.txt", "a") as f:
        #    f.write(
        #        "%s \n %s # %r # %r # %r \n" % (url_nifti, nifti_path, overwrite, verbose, os.path.exists(nifti_path)))
        #    f.close()
        nifti = try_to_request(
            self["scan"]["session"]["subject"]["project"].interface,
            url_nifti
        )
        # nifti = self["scan"]["session"]["subject"]["project"].interface.get(url_nifti, allow_redirects=True)
        nifti.raise_for_status()

        with open(os.path.join(complet_path, filename), 'wb') as nifti_file:
            nifti_file.write(nifti.content)

    def download_bids(self):
        pass

    def download_png(self, path_download, filename, overwrite=False, verbose=False):
        complet_path = (path_download + dict_path["path_download"](
            self["scan"]["session"]["subject"]["ID"],
            self["scan"]["session"]["ID"],
            self["scan"]["ID"],
            self["label"]
        )
                        )

        png_path = os.path.join(complet_path, filename)
        if not overwrite and os.path.exists(png_path):
            if verbose: print("png file already exist")
            return
        if verbose: print("Downloading png file...", flush=True)
        os.makedirs(complet_path, exist_ok=True)
        url_png = (self["scan"]["session"]["subject"]["project"].url_xnat
                   + dict_uri["files"](
                    self["scan"]["session"]["subject"]["project"]["ID"],
                    self["scan"]["session"]["subject"]["ID"],
                    self["scan"]["session"]["ID"],
                    self["scan"]["ID"],
                    self["label"]
                ).split("?")[0] + "/"
                   + filename
                   )

        # with open(path_download + "urls_descarga.txt", "a") as f:
        #    f.write("%s \n %s # %r # %r # %r \n" % (url_png, png_path, overwrite, verbose, os.path.exists(png_path)))
        #    f.close()
        png = try_to_request(
            self["scan"]["session"]["subject"]["project"].interface,
            url_png
        )
        # png = self["scan"]["session"]["subject"]["project"].interface.get(url_png, allow_redirects=True)
        png.raise_for_status()

        with open(os.path.join(complet_path, filename), 'wb') as png_file:
            png_file.write(png.content)

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

            if self["label"] == "DICOM" and bool_list_resources[1]:
                print(format_message(self.level_verbose + 1, self.level_tab + 1, f"{index} of {self['file_count']}"),
                      end="",
                      flush=True)
                # self.download_dicom(path_download, file_obj["Name"], overwrite=overwrite, verbose=verbose)

            if self["label"] == "SECONDARY" and bool_list_resources[2]:
                # self.download_dicom(path_download, file_obj["Name"], overwrite=overwrite, verbose=verbose)
                pass
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)
        print(format_message(self.level_verbose + 1, self.level_tab + 1, "\u001b[0K"), end="", flush=True)


class assessors(dict):
    def __init__(self, session, level_verbose, level_tab, **kwargs):
        super().__init__(**kwargs)
        self["session"] = session
        self.level_verbose = level_verbose
        self.level_tab = level_tab

    def get_list_assessors_resources(self, verbose):
        output = StringIO()
        if verbose:
            print(format_message(self.level_verbose, self.level_tab, f"assessor: {self['label']}"), flush=True)
        file_text = try_to_request(
            self["session"]["subject"]["project"].interface,
            self["session"]["subject"]["project"].url_xnat
            + dict_uri["assessors_resources"](
                self["session"]["subject"]["project"]["ID"],
                self["session"]["subject"]["ID"],
                self["session"]["ID"],
                self["ID"]
            )
        )
        file_text.raise_for_status()
        output.write(file_text.text)
        output.seek(0)
        reader = csv.DictReader(output)
        self.dict_resources = dict()
        for row in reader:
            self.dict_resources[row["xnat_abstractresource_id"]] = assessors_resources(
                self, self.level_verbose + 1, self.level_tab + 1, **row
            )
        output.close()

    def download(
            self, path_download,
            bool_list_resources=[False, False, True, False, False, False],
            overwrite=False,
            verbose=False
    ):

        # sys.exit(0)
        self.get_list_assessors_resources(verbose)
        for resource_obj in self.dict_resources.values():

            # sys.exit(1)
            try:
                resource_obj.download(
                    path_download,
                    bool_list_resources=bool_list_resources,
                    overwrite=overwrite, verbose=verbose)
            except requests.exceptions.RequestException as e:
                print("descarga fallida")
                traceback.print_exc()
                traceback.print_stack()
                sys.exit(1)
        print(format_message(self.level_verbose, self.level_tab, "\u001b[0K"), end="", flush=True)


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
            + dict_uri["roi_files"](
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
        complet_path = (path_download + dict_path["path_download_roi"](
            self["assessors"]["session"]["subject"]["ID"],
            self["assessors"]["session"]["ID"],
            self["assessors"]["ID"],
            self["xnat_abstractresource_id"]
        )
                        )

        roi_path = os.path.join(complet_path, filename)
        if not overwrite and os.path.exists(roi_path):
            if verbose: print(" roi file already exist")
            return
        if verbose: print(" Downloading png file...", flush=True)
        os.makedirs(complet_path, exist_ok=True)
        url_roi = (self["assessors"]["session"]["subject"]["project"].url_xnat
                   + dict_uri["files"](
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

        with open(os.path.join(complet_path, filename), 'wb') as roi_file:
            roi_file.write(roi.content)

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
