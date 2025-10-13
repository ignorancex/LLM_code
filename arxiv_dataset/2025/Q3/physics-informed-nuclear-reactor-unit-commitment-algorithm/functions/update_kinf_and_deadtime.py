import numpy as np
import pandas as pd

def update_kinf_and_deadtime(
    n,
    kinfTable_udt,
    gen_result_df,
    gen_df,
    MaxReacTableAP1000,
    MaxReacTableAP300,
    deadtime_matrix_ap1000,
    deadtime_matrix_ap300,
    terminate_loop_indicator,
    commit_at_len_T,
    D_points,
    mAP1000,
    mAP300,
    T,
    pmin,
    refuel_span
):
    """
    Update kinfTable, shutdown/refuel tracking, and deadtime values
    after solving a UC block for day n.
    Saves updated CSVs for use in the next iteration.
    """

    # Build MaxReacTable with resources
    MaxReacTableAP1000['resource'] = 'ap1000'
    MaxReacTableAP300['resource'] = 'ap300'
    MaxReacTable = pd.concat([MaxReacTableAP1000, MaxReacTableAP300], ignore_index=True)

    # Aggregate generation results
    gen_result_df_grouped = gen_result_df.copy(deep=True)
    gen_result_df_grouped["resource"] = gen_df["resource"]
    numeric_columns = [col for col in gen_result_df_grouped.columns if str(col).isnumeric()]
    gen_result_df_grouped['TotalGeneration'] = gen_result_df_grouped[numeric_columns].sum(axis=1)
    gen_result_df_grouped['r_id'] = gen_df['r_id']
    gen_result_df_grouped = pd.merge(
        gen_result_df_grouped,
        gen_df[['r_id', 'existing_cap_mw']],
        on='r_id',
        how='left'
    )

    # Compute alpha (capacity factor proxy)
    kinfTable_udt['alpha'] = kinfTable_udt['r_id'].map(
        gen_result_df_grouped.set_index('r_id')['TotalGeneration'] /
        (gen_result_df_grouped.set_index('r_id')['existing_cap_mw'] * len(T))
    )

    # Update keff by burnup
    kinfTable_udt.loc[kinfTable_udt['resource'] == 'ap1000', 'keff'] -= (
        mAP1000 * kinfTable_udt.loc[kinfTable_udt['resource'] == 'ap1000', 'alpha']
    )
    kinfTable_udt.loc[kinfTable_udt['resource'] == 'ap300', 'keff'] -= (
        mAP300 * kinfTable_udt.loc[kinfTable_udt['resource'] == 'ap300', 'alpha']
    )

    # Reactivity value
    kinfTable_udt['ReacValue'] = (1e5) * (kinfTable_udt['keff'] - 1) / kinfTable_udt['keff']

    # --- Compute p0 values based on MaxReacTable ---
    p0_values = []
    for _, row in kinfTable_udt.iterrows():
        resource = row['resource']
        reac_value = row['ReacValue']
        filtered_max_reac_table = MaxReacTable[MaxReacTable['resource'] == resource]

        differences = reac_value - filtered_max_reac_table['MaxReactivityXe']
        positive_differences = differences[differences >= 0]

        if not positive_differences.empty:
            min_diff_index = positive_differences.idxmin()
            corresponding_p0 = MaxReacTable.loc[min_diff_index, 'p0']
        else:
            corresponding_p0 = np.nan

        p0_values.append(corresponding_p0)

    # kinfTable_udt['p0'] = np.where(np.isnan(p0_values), 1, p0_values)
    kinfTable_udt['p0'] = np.where(np.isnan(p0_values), 1, np.maximum(p0_values, pmin))

    kinfTable_udt['nearest_k'] = np.nan

    # Initialize D_points if empty
    if D_points.empty:
        D_points['r_id'] = kinfTable_udt['r_id']
        D_points['deadtime_value'] = 0

    # --- Handle shutdown decrement ---
    terminate_loop_indicator['span_remaining'] = terminate_loop_indicator['span_remaining'].where(
        terminate_loop_indicator['span_remaining'] <= 0,
        terminate_loop_indicator['span_remaining'] - 1
    )

    shutdown_ids = terminate_loop_indicator.loc[terminate_loop_indicator['span_remaining'] > 0, 'r_id'].values
    kinfTable_udt.loc[
        kinfTable_udt['r_id'].isin(shutdown_ids),
        ['keff', 'p0', 'ReacValue', 'alpha']
    ] = 0.0

    r_ids_resetting = terminate_loop_indicator.loc[
        (terminate_loop_indicator['span_remaining'] == 0) &
        (terminate_loop_indicator['termination_condition'] == 1),
        'r_id'
    ]
    kinfTable_udt.loc[kinfTable_udt['r_id'].isin(r_ids_resetting), ['keff','p0','alpha']] = [1.205, 0, 0]
    kinfTable_udt.loc[kinfTable_udt['r_id'].isin(r_ids_resetting), 'ReacValue'] = 1e5*((1.205-1)/1.205)
    terminate_loop_indicator.loc[
        terminate_loop_indicator['r_id'].isin(r_ids_resetting),
        'termination_condition'
    ] = 0

    # --- Iterate to assign nearest_k and deadtime ---
    for idx, row in kinfTable_udt.iterrows():
        keff_value = row['keff']
        r_id_value = row['r_id']
        resource_value = row['resource']

        deadtime_matrix = deadtime_matrix_ap1000 if resource_value == 'ap1000' else deadtime_matrix_ap300

        k_values_less_than_keff = [
            float(col) for col in deadtime_matrix.columns if (col != 'p' and float(col) < keff_value)
        ]

        if not k_values_less_than_keff:
            last_column = next((col for col in reversed(deadtime_matrix.columns) if col != 'p'), None)
            nearest_k_value = float(last_column) if last_column else 1.0

            if terminate_loop_indicator.loc[
                terminate_loop_indicator['r_id'] == r_id_value, 'termination_condition'
            ].values[0] == 0:
                terminate_loop_indicator.loc[
                    terminate_loop_indicator['r_id'] == r_id_value,
                    ['termination_condition', 'span_remaining']
                ] = [1, refuel_span]

                kinfTable_udt.loc[kinfTable_udt['r_id']==r_id_value, ['keff','p0','ReacValue','alpha']] = 0.0
        else:
            nearest_k_value = max(k_values_less_than_keff)

        kinfTable_udt.loc[kinfTable_udt['r_id'] == r_id_value, 'nearest_k'] = nearest_k_value

        col_str = str(int(nearest_k_value)) if nearest_k_value.is_integer() else str(nearest_k_value)
        deadtime_value = deadtime_matrix[col_str].iloc[0]
        D_points.loc[D_points['r_id'] == r_id_value, 'deadtime_value'] = int(deadtime_value)

    # --- Commit update ---
    commit_at_len_T['termination_condition'] = (
        commit_at_len_T['generator']
        .map(terminate_loop_indicator.set_index('r_id')['termination_condition'])
        .fillna(0)
        .astype(int)
    )

    # --- Save for next iteration ---
    kinfTable_udt.to_csv(f'kinfTable_udt_{n+1}.csv')
    D_points.to_csv(f'D_points_{n+1}.csv')
    terminate_loop_indicator.to_csv(f'terminate_loop_indicator_{n+1}.csv')
    commit_at_len_T.to_csv(f'commit_at_len_T_{n+1}.csv')

    return kinfTable_udt, D_points, terminate_loop_indicator, commit_at_len_T
