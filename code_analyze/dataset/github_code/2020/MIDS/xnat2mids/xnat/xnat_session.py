###############################################################################
# Imports
###############################################################################
import csv
import getpass
import os
import progressbar
import requests
from io import StringIO
from xnat2mids.xnat.project import Project
from xnat2mids.xnat.request import try_to_request
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import reset_terminal
from xnat2mids.xnat.variables import dict_uris
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

class XnatSession:

    def __init__(self, url_xnat, user=None, password=None):
        self.url_xnat = url_xnat
        self.user = user
        self.password = password
        self.level_verbose = 0
        self.level_tab = 0
        self.project_list = []

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
        if verbose: print("list of projects to download: {}".format(", ".join(answer_list)), flush=True)
        return answer_list

    def download_projects(self,
                          path_download,
                          with_department=True,
                          bool_list_resources=[False, False, True, False, False, False],
                          overwrite=False,
                          verbose=False):
        self.get_projects(verbose)
        
        self.project_list = self.show_list_of_project(verbose)
        # clear console
        print(reset_terminal, end="", flush=True)
        print(format_message(self.level_verbose, self.level_tab, "Projects:"), end="", flush=True)
        print(format_message(self.level_verbose + 3, self.level_tab, "Subject:"), end="", flush=True)
        # move the cursor
        bar_project = progressbar.ProgressBar(maxval=len(self.project_list), prefix="\033[2;0H").start()
        bar_project.update(1)
        bar_project.update(0)

        for iter, key in enumerate(self.project_list):
            self.dict_projects[key].download(
                path_download,
                with_department=with_department,
                bool_list_resources=bool_list_resources,
                overwrite=overwrite,
                verbose=verbose
            )
            bar_project.update(iter + 1)
        bar_project.finish()
        format_message(self.level_verbose, self.level_tab, "")
        print(format_message(16, 0, ""), end="", flush=True)
