# -*- coding: utf-8 -*-
import pathlib

import click

help_settings = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=help_settings)
def base_command():
    pass


root_path = pathlib.Path.cwd().absolute()
"""Default destination for generated files"""
