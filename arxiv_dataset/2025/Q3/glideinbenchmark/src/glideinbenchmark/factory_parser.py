# SPDX-FileCopyrightText: 2023 Fermi Research Alliance, LLC
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def create_new_attr(root, value):
    # if we are removing the attribute, then return
    if value == "Remove":
        return 1
    # create new attrib
    new_attrib = ET.SubElement(root, "attr")
    new_attrib.attrib["name"] = "GlideinBenchmark"
    new_attrib.attrib["const"] = "False"
    new_attrib.attrib["glidein_publish"] = "True"
    new_attrib.attrib["job_publish"] = "False"
    new_attrib.attrib["parameter"] = "False"
    new_attrib.attrib["publish"] = "True"
    new_attrib.attrib["type"] = "string"
    new_attrib.attrib["value"] = value
    return 0


def update_attr(attrs, entry, run_bmk_value, entry_name="Global"):
    # if we are removing the attribute, then remove it and return
    if run_bmk_value == "Remove" and attrs is not None:
        entry.find("attrs").remove(attrs)
        print(f"Removed GlideinBenchmark from {entry_name}")
    # if the attribute exists, then update it; if not, create it
    else:
        if attrs is not None:
            attrs.attrib["value"] = run_bmk_value
            print(f"Updated {entry_name} with GlideinBenchmark={run_bmk_value}")
        else:
            if create_new_attr(entry.find("attrs"), run_bmk_value):
                print(
                    f"Cannot remove GlideinBenchmark from {entry_name} because it does not exist"
                )
            else:
                print(f"Created {entry_name} with GlideinBenchmark={run_bmk_value}")


def update_xml(xml_file, entry_names, run_bmk_value):
    try:
        # parse the xml file into an ElementTree objects
        tree = ET.parse(xml_file)
        # get the root of the tree
        root = tree.getroot()
        # go through the csv list of entries
        for entry_name in entry_names.split(","):
            # count to see if the entry name is valid
            count = 0
            # if the entry name is global, then we need to update the global entry
            if entry_name.lower() == "global":
                # get to the entry locations
                gentry = root.find("attrs")
                # gentry = root
                attrs = gentry.find(".//attr[@name='GlideinBenchmark']")
                update_attr(attrs, root, run_bmk_value)
            # otherwise, we need to update the entry with the given name
            else:
                # get to the entry locations
                entries = root.find("entries")
                # go through each entry and find the one with the given name
                for entry in entries.findall(".//entry"):
                    if entry.get("name") == entry_name:
                        count += 1
                        attrs = entry.find(".//attr[@name='GlideinBenchmark']")
                        update_attr(attrs, entry, run_bmk_value, entry_name)
            if count == 0:
                if entry_name.lower() != "global":
                    print(f"Entry {entry_name} does not exist")
        tree.write(xml_file)
    except ET.ElementTree.ParseError:
        print(f"Error parsing XML file: {xml_file}")
        exit(0)


def list_entries(xml_file):
    # parse the xml file into an ElementTree objects
    tree = ET.parse(xml_file)
    # get the root of the tree
    root = tree.getroot()
    # Look for all the possible entries
    entries = root.find("entries")
    # set a list to store all the entries in
    entries_list = []
    # go through each entry and find the one with the given name
    for entry in entries.findall(".//entry"):
        # print(entry.get("name"))
        entries_list.append(entry.get("name"))
    entries_list.append("global")
    return entries_list


def check_file_exists(xml_file):
    if not os.path.isfile(xml_file):
        print(f"File {xml_file} does not exist")
        return False
    return True


def list_entry_status(xml_file):
    # parse the xml file into an ElementTree objects
    tree = ET.parse(xml_file)
    # get the root of the tree
    root = tree.getroot()
    # Look for all the possible entries
    entries = root.find("entries")
    # Look for the global entry
    gentry = root.find("attrs")
    # set a dict to store all the entries in
    entries_dict = {}
    # # go through each entry and find the one with the given name
    for entry in entries.findall(".//entry"):
        entries_dict[entry.get("name")] = check_entry_status(entry)
    entries_dict["global"] = check_entry_status(gentry)
    return entries_dict


def check_entry_status(entry):
    attrs = entry.find(".//attr[@name='GlideinBenchmark']")
    if attrs is not None:
        if attrs.get("value") == "True":
            return "Enabled"
        elif attrs.get("value") == "False":
            return "Disabled"
    else:
        return "Removed"


def main():
    # parse the command line arguments
    parser = argparse.ArgumentParser(
        prog="GlideinBmk Factory Parser",
        description="Update the xml file with the given entry names and run_bmk_value",
    )
    parser.add_argument(
        "-d",
        "--disable",
        action="store_true",
        help="disable GlideinBenchmark for entry_names",
    )
    parser.add_argument(
        "-e",
        "--enable",
        action="store_true",
        help="enable GlideinBenchmark for entry_names",
    )
    parser.add_argument("-g", "--glob", action="store_true", help="add global entry")
    parser.add_argument(
        "-l", "--list", action="store_true", help="list all entries in the xml file"
    )
    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="remove GlideinBenchmark from entry_names",
    )
    parser.add_argument(
        "-s",
        "--status",
        action="store_true",
        help="check current status of GlideinBenchmark for entry_names",
    )
    parser.add_argument(
        "entry_names",
        type=str,
        help="comma separated list of entries to update",
        nargs="?",
    )
    parser.add_argument("file_name", type=str, help="xml file name")
    args = parser.parse_args()

    # get the xml file name
    xml_file = args.file_name

    # check if the file exists
    if check_file_exists(xml_file) == False:
        exit(0)
    # get the tree, root, and entries as a list
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # gentry = root.find('attrs')
    # entries = root.find('entries')
    # add global entry if -g is specified
    if args.glob:
        if args.entry_names:
            args.entry_names = args.entry_names + ",global"
        else:
            args.entry_names = "global"

    # list all the entries in the xml file if -l is specified
    if args.list:
        entries_list = list_entries(xml_file)
        print("Entries:")
        for entry in entries_list:
            print("  " + entry)
        if args.glob:
            print("  Global")
        exit(0)

    # check status on all entries specified if -s is specified
    if args.status:
        if args.entry_names:
            print("Status:")
            entries_dict = list_entry_status(xml_file)
            for entry_name in args.entry_names.split(","):
                if entry_name == "global":
                    print("  Global: " + entries_dict["global"])
                else:
                    print("  " + entry_name + ": " + entries_dict[entry_name])
            exit(0)
        else:
            print("Please specify entry_names")
            exit(1)
    # priority: remove > disable > enable
    if args.remove:
        run_bmk_value = "Remove"
    elif args.disable:
        run_bmk_value = "False"
    elif args.enable:
        run_bmk_value = "True"
    else:
        run_bmk_value = None

    if run_bmk_value is None:
        print(
            "Please provide option to enable (-e), disable (-d), or remove (-r) GlideinBenchmark"
        )
        sys.exit(1)
    update_xml(xml_file, args.entry_names, run_bmk_value)


if __name__ == "__main__":
    main()
