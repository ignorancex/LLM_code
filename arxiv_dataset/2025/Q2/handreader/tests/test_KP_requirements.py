import pkg_resources
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def parse_requirements(file_path):
    """Parse a requirements.txt file into a dictionary of {package: version}."""
    requirements = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip comments and empty lines
                if "==" in line:
                    package, version = line.split("==")
                    requirements[package.lower()] = version
    return requirements


def get_installed_packages():
    """Get installed packages and their versions."""
    installed_packages = {dist.key: dist.version for dist in pkg_resources.working_set}
    return installed_packages


def test_requirements_versions():
    """Test that all packages have the same version as in requirements.txt."""
    path = rootutils.find_root(search_from=__file__, indicator=".project-root")
    requirements_path = f"{path}/RGB_KP_reqs.txt"
    required_packages = parse_requirements(requirements_path)
    installed_packages = get_installed_packages()

    for package, required_version in required_packages.items():
        installed_version = installed_packages.get(package)
        assert installed_version == required_version, (
            f"Package '{package}' version mismatch: "
            f"required '{required_version}', installed '{installed_version}'"
        )
