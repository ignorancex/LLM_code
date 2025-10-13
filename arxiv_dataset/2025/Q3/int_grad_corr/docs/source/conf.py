# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys


sys.path.insert(0, "../")

# -- Project information -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "IGC"
copyright = "2024, Pierre Lelièvre"
author = "Pierre Lelièvre"

# -- General configuration -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

autoclass_content = "class"

napoleon_custom_sections = ["Attributes", "Methods"]

bibtex_bibfiles = ["biblio.bib"]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_favicon = "_static/plelievre_favicon.png"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/plelievre/int_grad_corr",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ]
}

# -- Options for link code extension -------------------------------------------


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    if filename == "igc":
        filename = "igc/igc"
    return f"https://github.com/plelievre/int_grad_corr/blob/main/{filename}.py"
