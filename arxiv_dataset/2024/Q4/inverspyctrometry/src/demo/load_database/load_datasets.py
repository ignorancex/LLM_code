from pprint import pprint

from src.interface.configuration import load_config


def main():
    db = load_config().database()

    for dataset_id in range(3):
        dataset_schema = db.datasets[dataset_id]
        pprint(dataset_schema)

        spectrum = db.dataset_spectrum(ds_id=dataset_id)
        pprint(spectrum.data.shape)
        pprint(spectrum.wavenumbers)
        pprint(spectrum.wavenumbers_unit)

        print()

    for dataset_id in range(3, 5):
        dataset_schema = db.datasets[dataset_id]
        pprint(dataset_schema)

        interferogram = db.dataset_interferogram(ds_id=dataset_id)
        pprint(interferogram.data.shape)
        pprint(interferogram.opds)
        pprint(interferogram.opds_unit)

        print()


if __name__ == "__main__":
    main()
