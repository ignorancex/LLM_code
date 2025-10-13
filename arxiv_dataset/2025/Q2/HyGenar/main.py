from packages.cli.app import app
from packages.utils.file import init_folders
from packages.utils.log import setup_logger

if __name__ == '__main__':
    init_folders()
    setup_logger()
    app()