import argparse

from {{cookiecutter.package_dir_name}}.version import __version__  # noqa


def main():
    parser = argparse.ArgumentParser(
        prog="{{ cookiecutter.project_name }}",
        description=(
            "{{ cookiecutter.project_short_description }}\n\n"
            "For more information, visit: "
            "https://github.com/{{ cookiecutter.github_username_or_orgname }}/{{ cookiecutter.github_repo_name }}/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"{{ cookiecutter.project_name }} {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
