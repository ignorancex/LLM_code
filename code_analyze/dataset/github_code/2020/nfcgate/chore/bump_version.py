#!/usr/bin/env python3
"""
Bumps the versionCode and versionName in the app/build.gradle file. This assumes version information is in semver format.

* Promotes the debug version to release version by removing -dev suffix.
* Increments the versionCode.
* Increments the debug version (versionNameOverride) by bumping patch part.
"""

import re
import semver
import argparse

def bump_version_part(version: semver.VersionInfo, part: str) -> semver.VersionInfo:
    match part:
        case "major":
            return version.bump_major()
        case "minor":
            return version.bump_minor()
        case "patch":
            return version.bump_patch()
        case _:
            raise ValueError(f"Invalid part: {part}")


# Set up argument parser
parser = argparse.ArgumentParser(description="Bumps the versionCode and versionName in the app/build.gradle file")
parser.add_argument("part", nargs='?', choices=["major", "minor", "patch"], default="patch", help="Part of the version to bump")
parser.add_argument("-y", "--yes", action="store_true", help="Disable interactive confirmation prompt")
args = parser.parse_args()

with open("app/build.gradle", "r") as file:
    contents = file.read()

# get current version information for debug and release
versionCode = re.search(r"versionCode (\d+)", contents).group(1)
versionName = re.search(r"versionName '(.+?)'", contents).group(1)
versionNameOverride = re.search(r"versionNameOverride = '(.+?)'", contents).group(1)

print(f"Current version: {versionCode} ({versionName})")
print(f"Current debug version: {versionNameOverride}")

# promote debug version to release version in file
new_versionName = semver.VersionInfo.parse(versionNameOverride).finalize_version()
contents = re.sub(r"versionName '(.+?)'", f"versionName '{new_versionName}'", contents)

# increment versionCode in file
new_versionCode = int(versionCode) + 1
contents = re.sub(r"versionCode (\d+)", f"versionCode {new_versionCode}", contents)

# increment new_versionName using semver for new debug version in file
new_versionName_debug = bump_version_part(new_versionName, args.part).replace(prerelease="dev")
contents = re.sub(r"versionNameOverride = '(.+?)'", f"versionNameOverride = '{new_versionName_debug}'", contents)

# print new version
print(f"New release version: {new_versionCode} ({new_versionName})")
print(f"New debug version: {new_versionName_debug}")

# interactive confirmation prompt
if not args.yes:
    confirm = input("Do you want to write these changes to app/build.gradle? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        exit()

# write to file
with open("app/build.gradle", "w") as file:
    file.write(contents)
