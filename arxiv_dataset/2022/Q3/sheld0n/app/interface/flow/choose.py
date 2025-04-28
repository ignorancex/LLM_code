#!/usr/bin/env python3

import os

choices_dir = "flow"
choices_exceptions = []

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    lib_filename = script_dir[:script_dir.find("interface") + len("interface")] + "/libchoose.py"
    # load lib
    with open(lib_filename, "rb") as lib_source_file:
        lib_code = compile(lib_source_file.read(), lib_filename, "exec")
    exec(lib_code)
    # run tests
    choose_file(choices_dir, choices_exceptions)
