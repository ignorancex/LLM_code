from typing import Callable

from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from tqdm import tqdm


def query(
    query_function: Callable[[str], str],
    source_ids: list[int | str],
    chunksize: int = 50000,
    progress: bool = True,
) -> Table:
    """
    Query the Gaia database for a list of source IDs.
    Queries are split into chunks of size 'chunksize'
    to avoid timeouts.

    Parameters
    ----------
    query_function : Callable[[str], str]
        A function that takes a string of source IDs and returns
        a ADQL query string, which is passed to the Gaia query service.
    source_ids : list[int | str]
        A list of source IDs to query.
    chunksize : int, optional
        The size of each chunk, by default 20000.
    progress : bool, optional
        If True, display a progress bar, by default True.

    Returns
    -------
    Table
        An astropy Table containing the results of the query.
    """

    # split the list of IDs into chunks
    chunks = [
        source_ids[i : i + chunksize] for i in range(0, len(source_ids) + 1, chunksize)
    ]

    results = []
    for chunk in tqdm(chunks, disable=not progress):
        # convert list to a format suitable for the SQL query
        formatted_ids = ",".join([str(id) for id in chunk])

        chunk_query = query_function(formatted_ids)

        job = Gaia.launch_job_async(chunk_query)
        chunk = job.get_results()

        results.append(chunk)

    results = vstack(results)
    return results
