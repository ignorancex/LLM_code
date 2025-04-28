import numpy as np

from src.common_utils.utils import convert_hertz_to_meter, convert_meter_units


def main():
    mapper = {
        "vis": {
            "freq": {
                "range": [790000000000000, 400000000000000],
                "unit": "hz",
            }
        },
        "nir": {
            "freq": {
                "range": [400000000000000, 214000000000000],
                "unit": "hz",
            }
        },
        "swir": {
            "freq": {
                "range": [214000000000000, 100000000000000],
                "unit": "hz",
            }
        },
        "mwir": {
            "freq": {
                "range": [100000000000000, 37000000000000],
                "unit": "hz",
            }
        },
        "lwir": {
            "freq": {
                "range": [37000000000000, 20000000000000],
                "unit": "hz",
            }
        },
        "fir": {
            "freq": {
                "range": [20000000000000, 300000000000],
                "unit": "hz",
            }
        },
        "custom": {
            "freq": {
                "range": [1400000000000, 200000000000],
                "unit": "hz",
            }
        },
    }

    spectrum = mapper["custom"]["freq"]
    freq = np.array(spectrum["range"])

    wl_unit = "nm"
    wl = convert_hertz_to_meter(values=freq, to_=wl_unit)

    wn_unit = "um"
    wn = 1 / convert_meter_units(values=wl[::-1], from_=wl_unit, to_=wn_unit)

    print(f"Freq: {freq} {spectrum['unit']}")
    print(f"Wavelength: {np.round(wl, decimals=0)} {wl_unit}")
    print(f"Wavenumber: {np.round(wn, decimals=5)} 1/{wn_unit}")


if __name__ == "__main__":
    main()
