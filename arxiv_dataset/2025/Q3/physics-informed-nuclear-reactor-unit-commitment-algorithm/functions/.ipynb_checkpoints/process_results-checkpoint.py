import pandas as pd

def process_results(df, gen_phs, loads, result, curtail, 
                               model_name, hour_index, n, output_prefix="transposed_df"):
    """
    Process the UC result DataFrame into a transposed, enriched format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw result DataFrame to transpose.
    gen_phs : pd.DataFrame
        Pumped hydro storage generation data (charge, discharge, soc).
    loads : object
        Object with attribute `.demand` representing demand profile.
    result : dict
        UC solver results containing 'NSE'.
    curtail : pd.DataFrame
        Curtailment information (curt, wind_curtail, solar_curtail, possible).
    model_name : str
        The name of the nuclear/fossil generator column in df.
    hour_index : list/array
        Hour indices for the horizon.
    n : int
        Current rolling day index.
    output_prefix : str, optional
        Prefix for CSV file output (default = "transposed_df").
    
    Returns
    -------
    transposed_df : pd.DataFrame
        Processed results DataFrame with additional metrics.
    """
    
    # Transpose and set column names
    transposed_df = df.transpose()
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df = transposed_df[1:]
    
    # Add derived columns
    transposed_df['actual_solar'] = transposed_df['solar_photovoltaic']
    transposed_df['actual_wind'] = transposed_df['onshore_wind_turbine']
    transposed_df["charge"] = gen_phs["charge"]
    transposed_df["discharge"] = gen_phs["discharge"]
    transposed_df["soc"] = gen_phs["soc"]
    transposed_df["demand"] = loads.demand
    transposed_df["NSE"] = result['NSE'].value
    
    transposed_df["net_supply"] = transposed_df[
        [model_name, "hydroelectric_pumped_storage", "solar_photovoltaic", 
         "onshore_wind_turbine", "NSE"]
    ].sum(axis=1)
    
    transposed_df["curtail"] = curtail["curt"]
    transposed_df['wind_curtail'] = curtail['wind_curtail']
    transposed_df['solar_curtail'] = curtail['solar_curtail']
    transposed_df['max_solar_possible'] = curtail['solar_possible']
    transposed_df['max_wind_possible'] = curtail['wind_possible']
    transposed_df["max_vre_possible"] = curtail["max_vre_possible"]
    
    # Add metadata
    transposed_df["Hour"] = hour_index
    transposed_df["n"] = n
    
    # Save for checkpointing
    outpath = f"{output_prefix}_{n}.csv"
    transposed_df.to_csv(outpath)
    
    return transposed_df
