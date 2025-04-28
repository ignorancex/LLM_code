from astropy.coordinates import SkyCoord
import astropy.units as u
from pyvo import registry


def ra_dec_to_galactic(ra, dec):
    """
    Convert right ascension and declination to galactic coordinates

    Args:
        ra: float. Right ascension in decimal degrees
        dec: float. Declination in decimal degrees

    Returns:
        float. Galactic longitude in degrees
        float. Galactic latitude in degrees
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs') # Create a SkyCoord object
    galactic = coord.galactic # Convert to galactic coordinates
    return galactic.l.degree, galactic.b.degree # Return the galactic coordinates


def mas_to_parsec(mas):
    """
    Convert milliarcseconds to parsecs

    Args:
        mas: float. Milliarcseconds

    Returns:
        float. Parsecs
    """
    return 1 / (mas / 1000)


def get_catalogue_data(catalogue, table_name, column_names):
    """
    Get data from a VizieR catalogue using the pyvo library
    Args:
        catalogue: str. The name of the VizieR catalogue. E.g: "J/AJ/156/84"
        table_name: str. The name of the table in the catalogue. E.g: "table3"
        column_names: list of str. The names of the columns to retrieve from the table. E.g: ["Nstars", "Age-CMD", "RAJ2000", "DEJ2000", "Plx"]. MUST be double quoted if the column name contains special characters or spaces.
    """
    catalogue_ivoid = f"ivo://CDS.VizieR/{catalogue}"
    voresource = registry.search(ivoid=catalogue_ivoid)[0]
    tables = voresource.get_tables()
    # We can also extract the tables names for later use
    tables_names = list(tables.keys())
    print("Tables in the catalogue: ", tables_names)
    table = catalogue + "/" + table_name
    quoted_column_names = [f'"{name}"' for name in column_names]  # Add double quotes around each column name
    adql_query = f"""
        SELECT {', '.join(quoted_column_names)}
        FROM "{table}"
    """
    tap_records = voresource.get_service("tap").run_sync(
        adql_query,
    )
    return tap_records