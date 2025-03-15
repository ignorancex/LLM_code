"""The automatic test definition file."""

import nox


@nox.session(name="no-credentials-test")
def run_test_fast(session):
    """Run pytest."""
    session.install(".[dev]")
    session.run("pytest", "-m", "not uses_credentials")


@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install(".[dev]")
    session.run("pytest")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src", "tests", "noxfile.py")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install(".[dev]")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "--no-warn-return-any",
        "--explicit-package-bases",
        "src",
        "tests",
    )


@nox.session(name="docs")
def docs(session):
    """Build docs."""
    session.install(".")
    session.install("sphinx-book-theme")
    session.install("sphinxcontrib-bibtex")
    session.run(
        "python",
        "-m",
        "sphinx",
        "-W",
        "-b",
        "html",
        "-d",
        "docs/build/doctrees",
        "docs/",
        "docs/_build/html",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", ".")
    session.run("black", ".")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install(".")
    session.install("pytest")
    session.install("coverage")
    try:
        session.run("coverage", "run", "--source=paper_crawler", "-m", "pytest")
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website."""
    session.run("rm", "-r", "htmlcov", external=True)


@nox.session(name="check-package")
def pyroma(session):
    """Run pyroma to check if the package is ok."""
    session.install("pyroma")
    session.run("pyroma", ".")
