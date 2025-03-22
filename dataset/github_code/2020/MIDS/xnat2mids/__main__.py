#!/usr/bin/env  python   #iniciaize enviroment
# -*- coding: utf-8 -*-.
u"""
Subdirección General de Sistemas de Información para la Salud

Centro de Excelencia e Innovación Tecnológica de Bioimagen de la Conselleria de Sanitat

http://ceib.san.gva.es/

María de la Iglesia Vayá -> delaiglesia_mar@gva.es or miglesia@cipf.es

Jose Manuel Saborit Torres -> jmsaborit@cipf.es

Jhon Jairo Saenz Gamboa ->jsaenz@cipf.es

Joaquim Ángel Montell Serrano -> jamontell@cipf.es

---

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as published by
the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/gpl-3.0.txt>.

---

Prerequisites

Python --version >= 3.8

Description:
    This application allow the user to download one project on XNAT and
    transform this project in MIDS format

    
"""
###############################################################################
# Imports
###############################################################################

import os
import argparse
from pathlib import Path
# serialize model to json

from xnat2mids.xnat.xnat_session import XnatSession
from xnat2mids.xnat.xnat_session import list_directory_xnat
#import mids_conversion
from xnat2mids.xnat.variables import types_files_xnat
from xnat2mids.xnat.variables import format_message
from xnat2mids.xnat.variables import reset_terminal
from xnat2mids.xnat.variables import dict_paths
from xnat2mids.xnat.variables import dict_uris
from xnat2mids.mids_conversion import create_directory_mids_v1
from xnat2mids.mids_conversion import create_tsvs


###############################################################################
# Functions
###############################################################################


def main():
    """
    This Fuction is de main programme
    """
    # Control of arguments of programme
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This sorfware allows the user to Download one project into XNAT platform
    and convert the XNAT images directory in MIDS directory.

    The aplication execution needs Python --version >= 3.6.7

    """
    )
    parser.add_argument('-w', '--web', type=str, default=None,
                        help='The URL page where XNAT is.')
    parser.add_argument('-u', '--user', type=str, default=None,
                        help="""The username to login in XNAT
                                    If not write a username, It logins
                                    as guest.""")
    parser.add_argument('-i ', '--input', type=str,
                        help="""the directory where the files will
                            be downloaded.""")
    parser.add_argument('-d ', '--debug-level', type=int, default=0,
                        help="""for execute partialy some functions:
                        \t + 0: no debug
                        \t + 1: only xnat download
                        \t + 2: xnat download + conversion to nifti
                        \t + 3: xnat download + conversion to nifti + classify check
                        """)
    # parser.add_argument('-p ', '--projects', nargs='*', default=None, type=str,
    #                     help="""The project name to download, if the project is
    #                         omitted,the aplication show all projects in xnat
    #                         to choice.""")
    parser.add_argument('-v ', '--verbose', default=False, action='store_true',
                        help="""Active Verbose""")
    parser.add_argument('-bp', '--body-part', dest="body_part", type=str,
                       help="""Specify which part of the body are 
                        in the dataset(REQUIRED if tag body part not exist in dicom images)""")
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help=""" Overwrite download files""")
    # parser.add_argument('-t', '--types', type=str, default="nr", nargs=1,
    #                     help="""Download types of MRI images
    #                         included in xnat
    #                         - s = snapshoot
    #                         - d = dicom + dicom metadata in folder NIFTI
    #                         - n = nifti or png if image is one slide (2D)
    #                         - r = Structural report
    #                         - m = Roi segmentation (Mask)
    #                         default = nr""")
    # save the arguments in varaibles
    args = parser.parse_args()
    # print(reset_terminal, end="", flush=True)
    # print(args)
    page = args.web
    user = args.user
    xnat_data_path = Path(args.input)
    mids_data_path = Path(args.input)
    debug_level = args.debug_level
    project_list = []
    verbose = args.verbose
    body_part = args.body_part
    #types = args.types[0]
    overwrite = args.overwrite
    # Comprobation if Xnat dowload can be execute
    if xnat_data_path and page:
        xnat_data_path.mkdir(exist_ok=True)
        xnat_sesion_object = XnatSession(page, user)
        with xnat_sesion_object as xnat_session:
            xnat_session.download_projects(
                xnat_data_path,
                with_department=True,
                bool_list_resources=[True for char in types_files_xnat],
                overwrite=overwrite,
                verbose=verbose
            )
        project_list = xnat_sesion_object.project_list
    # conditions to move xnat project to MIDS project
    if debug_level == 1: return
        # if project_list is None, the projects in the folder xnat must be
        # chosen
    
    project_paths = [dirs for dirs in xnat_data_path.iterdir()]
    project_names = [path_.name for path_ in project_paths]
    
    project_list = list_directory_xnat(project_names) if not project_list else project_list
    
    mids_data_path.mkdir(exist_ok=True)
    # for each project choice
    for xnat_project in project_list:
        xnat_data_path = xnat_data_path.joinpath(xnat_project,"sourcedata")
        if not xnat_data_path.exists(): 
            raise FileNotFoundError(f'No folder exists at the location specified in {xnat_data_path}') 
        mids_data_path = mids_data_path.joinpath(xnat_project)
        if debug_level < 4:
            print("MIDS are generating...")
            create_directory_mids_v1(
                xnat_data_path,
                mids_data_path,
                body_part,
                debug_level
            )

        print("tsvs are generating...")
        create_tsvs(xnat_data_path, mids_data_path, body_part)

        # print("scan tsv are generating...")
        # MIDS_funtions.create_scans_tsv(
        #     os.path.join(mids_data_path, xnat_project)
        # )
        #
        # print("sessions tsv are generating...")
        # MIDS_funtions.create_sessions_tsv(
        #     os.path.join(xnat_data_path, xnat_project),
        #     mids_data_path
        # )

if __name__ == "__main__":
    main()
