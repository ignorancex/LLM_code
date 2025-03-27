import os
import re
import subprocess


def resolve_package_urls(string: str, prefix=""):
    substitutions = {}
    for package_expr in set(re.findall(r"package://.*?/", string)):
        package_name = package_expr.split("/")[-2]
        package_path = subprocess.check_output(["rospack", "find", package_name])
        substitutions[package_expr] = prefix + package_path.decode("utf-8").strip() + "/"
    for package_expr, package_path in substitutions.items():
        print("Patching {}: {}".format(package_expr, package_path))
        string = string.replace(package_expr, package_path)
    return string


def resolve_urdf(urdf_filepath):
    if os.path.splitext(urdf_filepath)[-1] == ".xacro":
        urdf_str = subprocess.check_output(["xacro", urdf_filepath])
    else:
        with open(urdf_filepath) as urdf_file:
            urdf_str = urdf_file.read()
    return resolve_package_urls(urdf_str)
