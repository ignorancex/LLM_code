from scikit_package.utils import cookie

SKPKG_GITHUB_URL = "https://github.com/scikit-package/scikit-package"


def package(args):
    """Run the cookiecutter template for creating a package.

    By default, checkout the latest release tag from the relevant GitHub
    repository.
    """
    subcmd = args.subcommand
    if subcmd == "workspace":
        cookie.run(f"{SKPKG_GITHUB_URL}-workspace")
    elif subcmd == "system":
        cookie.run(f"{SKPKG_GITHUB_URL}-system")
    elif subcmd == "public":
        cookie.run(SKPKG_GITHUB_URL)
    elif subcmd == "conda-forge":
        cookie.run(f"{SKPKG_GITHUB_URL}-conda-forge")
    elif subcmd == "manuscript":
        cookie.run(f"{SKPKG_GITHUB_URL}-manuscript")
