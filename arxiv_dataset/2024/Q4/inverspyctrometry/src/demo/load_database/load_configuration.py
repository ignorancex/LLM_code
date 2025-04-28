from pprint import pprint

from src.interface.configuration import load_config


def main():
    config = load_config()

    path = config.directory_paths
    pprint(path)
    print()

    db_paths = config.database_paths
    pprint(db_paths)
    print()

    db = config.database()
    pprint(db.characterizations)
    pprint(db.datasets)
    pprint(db.experiments)
    pprint(db.interferometers)
    pprint(db.inversion_protocols)
    pprint(db.noise_levels)


if __name__ == "__main__":
    main()
