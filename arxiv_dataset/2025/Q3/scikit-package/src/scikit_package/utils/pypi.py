import requests
from packaging.version import parse as parse_version


def check_pypi_package_exists(package):
    """Check if a package exists on PyPI and print the latest
    version."""
    response = requests.get(f"https://pypi.org/pypi/{package}/json")
    if response.status_code == 200:
        data = response.json()
        version = data["info"]["version"]
        print(f"> {package} is available on PyPI (latest version: {version}).")
    else:
        raise ValueError(
            f"{package} is not found on PyPI. "
            "`package create conda-forge` currently only supports pulling "
            "information about the package from PyPI. Please ensure your "
            "package is uploaded to PyPI. If you want to upload package "
            "to conda-forge sourced from GitHub instead, "
            "please update the conda-forge recipe by hand.",
        )


def get_pypi_version_sha(package, count=1):
    """Fetch the latest stable versions of the package and their
    SHA256."""
    response = requests.get(f"https://pypi.org/pypi/{package}/json")
    if response.status_code == 200:
        data = response.json()
        all_versions = [
            v
            for v in data["releases"].keys()
            if not parse_version(v).is_prerelease
        ]
        sorted_versions = sorted(
            all_versions, key=parse_version, reverse=True
        )[:count]
        version_info = {}
        for version in sorted_versions:
            files = data["releases"][version]
            for file in files:
                if file["packagetype"] == "sdist":
                    version_info[version] = file["digests"]["sha256"]
                    break
        return version_info
    else:
        raise ValueError(
            f"No matching package found for {package} on PyPI. "
            "Please check the name at https://pypi.org/project/"
        )
