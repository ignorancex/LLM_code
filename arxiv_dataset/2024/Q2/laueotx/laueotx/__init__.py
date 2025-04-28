import click
# from laueotx.apps.fakeapp import main as fakeapp
from .apps import realdata # noqa

@click.group()
def main():
    """main...
    """
    pass

# main.add_command(fakeapp)
main.add_command(realdata.main)

if __name__ == "__main__":
    main()