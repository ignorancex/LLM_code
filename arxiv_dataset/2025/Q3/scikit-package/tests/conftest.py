import json
from pathlib import Path

import pytest

files_in_old_project = {
    ".git/COMMIT_EDITMSG": """
skpkg: last commit message in skpkg-package
""",
    "docs/source/tutorial.rst": """
The tutorial for skpkg-package.
""",
    "README.rst": """
|Icon| |title|_
===============

.. |title| replace:: title of README.rst in skpkg-package
""",
    "docs/source/index.rst": """
#######
|title|
#######

.. |title| replace:: title of skpkg-package documentation
""",
}
files_in_new_project = {
    ".git/COMMIT_EDITMSG": """
The file already exists in the new project
skpkg: last commit message in skpkg-package
""",
    "docs/source/tutorial.rst": """
The file already exists in the new project
The tutorial for skpkg-package.
""",
}


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    old_package_dir = base_dir / "package-dir"
    for file_name, file_content in files_in_old_project.items():
        file_path = old_package_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_content)
    empty_source_dir = base_dir / "empty-source-dir"
    empty_source_dir.mkdir()
    target_dir_inside_old_package_dir = (
        old_package_dir / "target-dir-inside-package-dir"
    )
    for file_name, file_content in files_in_new_project.items():
        file_path = target_dir_inside_old_package_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_content)
    target_dir = base_dir / "target-dir"
    for file_name, file_content in files_in_new_project.items():
        file_path = target_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_content)

    yield tmp_path
