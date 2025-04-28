from pprint import pprint

from src.interface.configuration import load_config


def main():
    db = load_config().database()

    experiment_id = 0

    experiment = db.experiments[experiment_id]
    pprint(dict(experiment))


if __name__ == "__main__":
    main()
