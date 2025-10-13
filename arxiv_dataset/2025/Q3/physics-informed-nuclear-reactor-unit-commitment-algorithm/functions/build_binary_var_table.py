import pandas as pd

def build_binary_var_table(df, hour_index, name, n, output_prefix=""):
    """
    Build a clean result table (GEN, SHUT, START, COMMIT) from solver output.
    
    Parameters
    ----------
    df : pd.DataFrame
        The raw result DataFrame (wide form with generators in rows).
    hour_index : list/array
        Hour indices for the horizon.
    name : str
        Name of the table (e.g., "GEN", "SHUT", "START", "COMMIT").
    n : int
        Current day index for file naming.
    output_prefix : str, optional
        Directory or prefix for saving (default empty = current folder).
    
    Returns
    -------
    result_df : pd.DataFrame
        Processed table with 'hour' column first and index reset.
    """
    result_df = df.T
    result_df.columns = df.index
    result_df['hour'] = hour_index
    result_df = result_df[['hour'] + list(result_df.columns[:-1])]
    result_df.reset_index(drop=True, inplace=True)
    
    # Save
    outpath = f"{output_prefix}{name}_{n}.csv"
    result_df.to_csv(outpath)
    
    return result_df
