# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'zebradm'
copyright = '2025, Sebastian Sassi'
author = 'Sebastian Sassi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

import sys
print(sys.executable)

extensions = ['breathe']

breathe_projects = {project: '_build/xml'}
breathe_default_project = project

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------

project_name = 'ZebraDM'

latex_engine = 'lualatex'
latex_documents = [(master_doc, f'{project}_doc.tex', project_name, author, 'manual', False)]
latex_toplevel_sectioning = 'section'
latex_elements = {
    'papersize': 'a4paper',
    'fontpkg': r'''
\usepackage{unicode-math}
\setmainfont{STIXTwoText}[BoldFont=Stix Two Text Semibold]
\setsansfont{OpenSans}
\setmonofont{NotoSansMono}[Scale=0.91, BoldFont=Noto Sans Mono Bold]
\setmathfont{StixTwoMath}[StylisticSet=1]
    ''',
    'preamble': r'''
\newcommand{\subtitle}{Zernike-based Radon transforms}
\newcommand{\DUrolek}[1]{\textbf{\texttt{#1}}}
\newcommand{\DUrolekt}[1]{\texttt{#1}}
\newcommand{\DUrolen}[1]{\texttt{#1}}
\newcommand{\DUrolep}[1]{\texttt{#1}}
''',
    'extrapackages': r'''
\usepackage{microtype}
''',
    'releasename': 'Version',
    'maketitle': r'''
\makeatletter
\begin{titlepage}
\begin{center}
\vspace*{18pc}
{\Huge\textsf{\textbf{\texttt{\@title}}}}\\[2pc]
{\Large\textsf{\textbf{\subtitle}}}\\[2pc]
{\large\textsf{\@author}}
\vfill
{\normalsize\textsf{\texttt{\py@release\releaseinfo}\hfill\@date}}
\end{center}
\end{titlepage}
\makeatother
'''
}
