import pandas as pd

def init_state(G, G_thermal, kinfTable_udt, D_points, start_day):
    """
    Initialize all required DataFrames and save the starting state to CSV.
    """
    # Master tracking frames
    state = {
        "master_df": pd.DataFrame(),
        "GEN_df": pd.DataFrame(),
        "SHUT_df": pd.DataFrame(),
        "START_df": pd.DataFrame(),
        "RESUP_df": pd.DataFrame(),
        "RESDN_df": pd.DataFrame(),
        "SLOWRES_df": pd.DataFrame(),
        "master_shut_df": pd.DataFrame(),
        "master_commit": pd.DataFrame(),
        "master_kinfTable": pd.DataFrame(),
        "master_commit_at_len_T": pd.DataFrame(),
        "run_details": []
    }

    # Remaining downtime tracker
    remaining_downtime = pd.DataFrame({
        "generator": G_thermal,
        "remaining_downtime": 0
    })

    # Commitment state tracker
    commit_at_len_T = pd.DataFrame({
        "generator": G,
        "commitment": 0,                # initialize OFF
        "soc_transfer": 0,
        "termination_condition": 0
    })
    # Force G_thermal to ON if desired
    commit_at_len_T.loc[commit_at_len_T['generator'].isin(G_thermal), 'commitment'] = 0

    # Termination tracker
    terminate_loop_indicator = pd.DataFrame({
        "r_id": kinfTable_udt['r_id'],
        "termination_condition": 0,
        "span_remaining": 0
    })

    # === Save to CSV for rolling horizon restart ===
    remaining_downtime.to_csv(f'remaining_downtime_{start_day}.csv')
    commit_at_len_T.to_csv(f'commit_at_len_T_{start_day}.csv')
    kinfTable_udt.to_csv(f'kinfTable_udt_{start_day}.csv')
    D_points.to_csv(f'D_points_{start_day}.csv')
    terminate_loop_indicator.to_csv(f'terminate_loop_indicator_{start_day}.csv')

    state.update({
        "remaining_downtime": remaining_downtime,
        "commit_at_len_T": commit_at_len_T,
        "terminate_loop_indicator": terminate_loop_indicator,
        "terminate_loop": False
    })

    return state
