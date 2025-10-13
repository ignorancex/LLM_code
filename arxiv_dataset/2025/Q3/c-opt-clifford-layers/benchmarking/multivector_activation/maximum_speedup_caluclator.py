import pandas as pd
import os

def calculate_max_speedups():

    df = pd.read_csv("multivector_bench_x86_Max.csv")

    required_cols = ['Version', 'Mode', 'K', 'FLOP_per_cycle', 'n', 'I', 'B', 'C']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain the columns: {', '.join(required_cols)}")
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Missing columns: {missing}")
        return

    K_values_to_analyze = sorted(df['K'].unique()) 
    if 4 not in K_values_to_analyze: K_values_to_analyze.append(4)
    if 8 not in K_values_to_analyze: K_values_to_analyze.append(8)
    K_values_to_analyze = sorted(list(set(K_values_to_analyze).intersection([4,8])))

    modes_to_analyze = ['Linear', 'Sum', 'Mean']

    config_cols = ['Mode', 'n', 'I', 'B', 'C', 'K']

    print("Maximum Speedup (Opt5 FLOP_per_cycle / Baseline FLOP_per_cycle):\n")

    for K_val in K_values_to_analyze:
        if K_val not in [4, 8]: 
            continue

        print(f"--- For K = {K_val} ---")
        for mode_name in modes_to_analyze:

            df_filtered = df[(df['K'] == K_val) & (df['Mode'] == mode_name)]

            if df_filtered.empty:
                print(f"  Mode: {mode_name:<7} - No data found for K={K_val} and Mode={mode_name}")
                continue

            baseline_data = df_filtered[df_filtered['Version'] == 'Baseline'].set_index(config_cols)
            opt5_data = df_filtered[df_filtered['Version'] == 'Opt5'].set_index(config_cols)

            if baseline_data.empty or opt5_data.empty:
                print(f"  Mode: {mode_name:<7} - Missing Baseline or Opt5 data for K={K_val} and Mode={mode_name}")
                continue

            merged_data = baseline_data[['FLOP_per_cycle']].join(
                opt5_data[['FLOP_per_cycle']],
                lsuffix='_baseline',
                rsuffix='_opt5',
                how='inner' 
            )

            if merged_data.empty:
                print(f"  Mode: {mode_name:<7} - No matching configurations between Baseline and Opt5.")
                continue

            valid_baseline_flops = merged_data['FLOP_per_cycle_baseline'] > 0

            if not valid_baseline_flops.any():
                print(f"  Mode: {mode_name:<7} - All Baseline FLOP_per_cycle are zero or non-positive.")
                continue

            merged_data['Speedup'] = 0.0 
            merged_data.loc[valid_baseline_flops, 'Speedup'] = \
                merged_data.loc[valid_baseline_flops, 'FLOP_per_cycle_opt5'] / \
                merged_data.loc[valid_baseline_flops, 'FLOP_per_cycle_baseline']

            max_speedup = merged_data['Speedup'].max()

            if pd.isna(max_speedup):
                 print(f"  Mode: {mode_name:<7} - Could not calculate max speedup (NaN result).")
            else:
                print(f"  Mode: {mode_name:<7} - Max Speedup: {max_speedup:.2f}x")
        print("-" * 20)

if __name__ == "__main__":
    calculate_max_speedups()