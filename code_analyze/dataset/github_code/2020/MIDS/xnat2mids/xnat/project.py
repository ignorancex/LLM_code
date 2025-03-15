import csv
import os
import progressbar
from io import StringIO
from xnat2mids.xnat.subject import Subject
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import dict_uris
from xnat2mids.xnat.request import try_to_request
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
            print(format_message(self.level_verbose, self.level_tab, "Project: {}".format(self["secondary_ID"])), flush=True)
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
        path_download = path_download.joinpath(self["ID"], "sourcedata")
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