import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.config import INDICATORS, TRAINED_MODEL_DIR

# Define helper functions
def calculate_mdd(asset_values):
    """
    Calculate the Maximum Drawdown (MDD) of a portfolio.
    """
    running_max = asset_values.cummax()
    drawdown = (asset_values - running_max) / running_max
    mdd = drawdown.min() * 100  # Convert to percentage
    return mdd

def calculate_sharpe_ratio(asset_values, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio of a portfolio.
    """
    # Calculate daily returns
    returns = asset_values.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days
    if excess_returns.std() == 0:
        return 0.0
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
    return sharpe_ratio

# Load data
train = pd.read_csv('train_data.csv')
trade = pd.read_csv('test_data.csv')

# Preprocess data
train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

algorithms = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']

# Define a mapping for better legend labels
label_mapping = {
    'DT_LoRA_GPT2': 'DT-LoRA-GPT2',
    'DT_LoRA_Random_Weight_GPT2': 'DT-LoRA-Random-GPT2',
    'CQL': 'Conservative Q-Learning',
    'IQL': 'Implicit Q-Learning',
    'BC': 'Behavior Cloning',
    'A2C': 'A2C',
    'DDPG': 'DDPG',
    'PPO': 'PPO',
    'TD3': 'TD3',
    'SAC': 'SAC',
    'DJIA': 'Dow Jones Index'
}

for current_algo in algorithms:
    # Reset all algorithms to False
    if_using_a2c = False
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = False

    # Set the current algorithm to True
    if current_algo == 'a2c':
        if_using_a2c = True
    elif current_algo == 'ddpg':
        if_using_ddpg = True
    elif current_algo == 'ppo':
        if_using_ppo = True
    elif current_algo == 'td3':
        if_using_td3 = True
    elif current_algo == 'sac':
        if_using_sac = True

    # Reset algos_included for each iteration
    algos_included = ''

    print(f"Running with {current_algo.upper()} set to True")

    # Load trained models
    trained_a2c = A2C.load(os.path.join(TRAINED_MODEL_DIR, "agent_a2c")) if if_using_a2c else None
    trained_ddpg = DDPG.load(os.path.join(TRAINED_MODEL_DIR, "agent_ddpg")) if if_using_ddpg else None
    trained_ppo = PPO.load(os.path.join(TRAINED_MODEL_DIR, "agent_ppo")) if if_using_ppo else None
    trained_td3 = TD3.load(os.path.join(TRAINED_MODEL_DIR, "agent_td3")) if if_using_td3 else None
    trained_sac = SAC.load(os.path.join(TRAINED_MODEL_DIR, "agent_sac")) if if_using_sac else None

    # Define environment parameters
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    # Initialize trading environment
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    # Predict using trained models
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_a2c, 
        environment=e_trade_gym) if if_using_a2c else (None, None)

    df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
        model=trained_ddpg,
        environment=e_trade_gym) if if_using_ddpg else (None, None)

    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
        model=trained_td3,
        environment=e_trade_gym) if if_using_td3 else (None, None)

    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym) if if_using_sac else (None, None)

    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo, environment=e_trade_gym) if if_using_ppo else (None, None)

    # Set indices for result merging
    df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0]) if if_using_a2c else None
    df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0]) if if_using_ddpg else None
    df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0]) if if_using_ppo else None
    df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0]) if if_using_td3 else None
    df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0]) if if_using_sac else None

    # Merge results
    result = pd.DataFrame()
    if if_using_a2c: 
        algos_included += '_a2c'
        df_result_a2c.columns = ['A2C_' + col for col in df_result_a2c.columns]
        result = pd.merge(result, df_result_a2c, how='outer', left_index=True, right_index=True)
    if if_using_ddpg:
        algos_included += '_ddpg'
        df_result_ddpg.columns = ['DDPG_' + col for col in df_result_ddpg.columns]
        result = pd.merge(result, df_result_ddpg, how='outer', left_index=True, right_index=True)
    if if_using_td3: 
        algos_included += '_td3'
        df_result_td3.columns = ['TD3_' + col for col in df_result_td3.columns]
        result = pd.merge(result, df_result_td3, how='outer', left_index=True, right_index=True)
    if if_using_sac: 
        algos_included += '_sac'
        df_result_sac.columns = ['SAC_' + col for col in df_result_sac.columns]
        result = pd.merge(result, df_result_sac, how='outer', left_index=True, right_index=True)
    if if_using_ppo:
        algos_included += '_ppo'
        df_result_ppo.columns = ['PPO_' + col for col in df_result_ppo.columns]
        result = pd.merge(result, df_result_ppo, how='outer', left_index=True, right_index=True)

    # Function to load pickle files
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # Get all directories under ./checkpoints/
    checkpoint_dirs = [d for d in os.listdir('./checkpoints') if os.path.isdir(os.path.join('./checkpoints', d))]

    # Process each checkpoint directory
    for dir_name in checkpoint_dirs:
        dir_path = os.path.join('./checkpoints', dir_name)
        pkl_files = [f for f in os.listdir(dir_path) if f.startswith('total_asset_value_change_test') and f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            file_path = os.path.join(dir_path, pkl_file)
            data = load_pickle(file_path)
            data = data[:335]  # Limit data to first 335 points
            
            # Validate data length
            if len(data) != len(result.index):
                print(f"Warning: Data length mismatch for {pkl_file} in {dir_name}. Expected {len(result.index)}, got {len(data)}. Skipping this file.")
                continue
            
            # Use directory name and pkl file name (without extension) as column name
            column_name = f'{dir_name}_{os.path.splitext(pkl_file)[0]}'
            
            df = pd.DataFrame(data, columns=[column_name])
            df.index = result.index  # Assuming the dates align with the existing result dataframe
            result = pd.merge(result, df, how='outer', left_index=True, right_index=True)

    # Create column names list with better formatting
    col_name = []
    if if_using_a2c: col_name.append('A2C')
    if if_using_ddpg: col_name.append('DDPG')
    if if_using_td3: col_name.append('TD3')
    if if_using_sac: col_name.append('SAC')
    if if_using_ppo: col_name.append('PPO')
    col_name.extend(result.columns[len(col_name):])  # Add all other column names
    result.columns = col_name

    # Define test period
    TEST_START_DATE = '2020-07-01'
    TEST_END_DATE = '2021-10-29'

    # Fetch DJIA data for the test period
    df_dji = YahooDownloader(start_date=TEST_START_DATE,
                             end_date=TEST_END_DATE,
                             ticker_list=['dji']).fetch_data()

    df_dji = df_dji[['date','close']]
    fst_day = df_dji['close'].iloc[0]
    dji = pd.DataFrame({
        'DJIA': df_dji['close'].div(fst_day).mul(1000000)
    }, index=df_dji['date'])

    # Merge DJIA data using inner join to ensure alignment
    result = pd.merge(result, dji, how='inner', left_index=True, right_index=True).fillna(method='bfill')

    # Control variables
    include_ensemble = False  # Set to False to exclude ensemble experiments
    exclude_algo_experiments = True  # Set to True to exclude individual algo experiments based on if_using_... flags

    if include_ensemble:
        algos_included += '_ensemble'

    # Function to extract experiment name from directory name
    def extract_experiment_name(dir_name):
        # This regex matches everything up to the last underscore and number
        match = re.match(r'(.+)_\d+$', dir_name)
        if match:
            return match.group(1)
        return dir_name

    # Function to check if an experiment should be included
    def should_include_experiment(exp_name):
        if not include_ensemble and 'ensemble' in exp_name.lower():
            return False
        if exclude_algo_experiments:
            algo_flags = {
                'a2c': if_using_a2c,
                'ddpg': if_using_ddpg,
                'ppo': if_using_ppo,
                'td3': if_using_td3,
                'sac': if_using_sac
            }
            for algo, flag in algo_flags.items():
                if algo in exp_name.lower() and not flag:
                    return False
        return True

    # Group similar experiments
    experiment_groups = {}
    for column in result.columns:
        if column not in ['A2C', 'DDPG', 'TD3', 'SAC', 'PPO', 'DJIA']:
            exp_name = extract_experiment_name(column.split('_total_asset_value_change_test')[0])
            if should_include_experiment(exp_name):
                if exp_name not in experiment_groups:
                    experiment_groups[exp_name] = []
                experiment_groups[exp_name].append(column)

    # Initialize a dictionary to store metrics for comparison
    metrics_dict = {
        'Method': [],
        'Cumulative Return Mean (%)': [],
        'Cumulative Return Std (%)': [],
        'MDD Mean (%)': [],
        'MDD Std (%)': [],
        'Sharpe Ratio Mean': [],
        'Sharpe Ratio Std': []
    }

    # Calculate metrics for each experiment group
    experiment_stats = {}
    for exp_name, columns in experiment_groups.items():
        exp_data = result[columns].dropna()
        
        if exp_data.empty:
            print(f"Warning: No valid data for experiment '{exp_name}'. Skipping metrics calculation.")
            continue
        
        # Cumulative Return: (Final - Initial) / Initial * 100 for each run
        cumulative_returns = (exp_data.iloc[-1] - exp_data.iloc[0]) / exp_data.iloc[0] * 100
        
        # Handle potential division by zero or invalid calculations
        cumulative_returns = cumulative_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if cumulative_returns.empty:
            print(f"Warning: No valid cumulative returns for experiment '{exp_name}'. Skipping metrics calculation.")
            continue
        
        cumulative_return_mean = cumulative_returns.mean()
        cumulative_return_std = cumulative_returns.std()

        # MDD: Calculate MDD for each run
        mdd_values = []
        for col in columns:
            asset_values = result[col].dropna()
            if asset_values.empty:
                continue
            mdd_run = calculate_mdd(asset_values)
            mdd_values.append(mdd_run)
        
        if not mdd_values:
            print(f"Warning: No valid MDD values for experiment '{exp_name}'. Skipping MDD calculation.")
            mdd_mean = np.nan
            mdd_std = np.nan
        else:
            mdd_mean = np.mean(mdd_values)
            mdd_std = np.std(mdd_values)

        # Sharpe Ratio: Calculate Sharpe for each run
        sharpe_ratios = []
        for col in columns:
            asset_values = result[col].dropna()
            if asset_values.empty:
                continue
            sharpe_run = calculate_sharpe_ratio(asset_values)
            sharpe_ratios.append(sharpe_run)
        
        if not sharpe_ratios:
            print(f"Warning: No valid Sharpe Ratios for experiment '{exp_name}'. Skipping Sharpe Ratio calculation.")
            sharpe_mean = np.nan
            sharpe_std = np.nan
        else:
            sharpe_mean = np.mean(sharpe_ratios)
            sharpe_std = np.std(sharpe_ratios)

        # Append to metrics_dict with mapped label
        mapped_exp_name = label_mapping.get(exp_name, exp_name)
        metrics_dict['Method'].append(mapped_exp_name)
        metrics_dict['Cumulative Return Mean (%)'].append(cumulative_return_mean)
        metrics_dict['Cumulative Return Std (%)'].append(cumulative_return_std)
        metrics_dict['MDD Mean (%)'].append(mdd_mean)
        metrics_dict['MDD Std (%)'].append(mdd_std)
        metrics_dict['Sharpe Ratio Mean'].append(sharpe_mean)
        metrics_dict['Sharpe Ratio Std'].append(sharpe_std)

        # Store in experiment_stats for plotting
        experiment_stats[mapped_exp_name] = {'mean': exp_data.mean(axis=1), 'std': exp_data.std(axis=1)}

    # Calculate metrics for individual algorithms (A2C, DDPG, TD3, SAC, PPO)
    individual_algos = ['A2C', 'DDPG', 'TD3', 'SAC', 'PPO']
    for algo in individual_algos:
        if algo in result.columns:
            # Check if this algorithm is already part of experiment_groups
            if label_mapping.get(algo, algo) in experiment_stats:
                print(f"Info: '{algo}' is already included in experiment groups. Skipping individual plotting to avoid duplication.")
                continue  # Skip to prevent duplicate plotting

            asset_values = result[algo].dropna()
            if asset_values.empty:
                print(f"Warning: No valid asset values for individual algorithm '{algo}'. Skipping metrics calculation.")
                continue
            # Cumulative Return
            cum_ret = (asset_values.iloc[-1] - asset_values.iloc[0]) / asset_values.iloc[0] * 100
            # Handle potential division by zero or invalid calculations
            if np.isinf(cum_ret) or np.isnan(cum_ret):
                cum_ret = np.nan
            # MDD
            mdd = calculate_mdd(asset_values)
            # Sharpe Ratio
            sharpe = calculate_sharpe_ratio(asset_values)
            # Append to metrics_dict with mapped label
            mapped_algo = label_mapping.get(algo, algo)
            metrics_dict['Method'].append(mapped_algo)
            metrics_dict['Cumulative Return Mean (%)'].append(cum_ret)
            metrics_dict['Cumulative Return Std (%)'].append(0.00)  # Single run, std is 0
            metrics_dict['MDD Mean (%)'].append(mdd)
            metrics_dict['MDD Std (%)'].append(0.00)  # Single run, std is 0
            metrics_dict['Sharpe Ratio Mean'].append(sharpe)
            metrics_dict['Sharpe Ratio Std'].append(0.00)  # Single run, std is 0

            # Store in experiment_stats for plotting
            experiment_stats[mapped_algo] = {'mean': asset_values, 'std': pd.Series([0]*len(asset_values), index=asset_values.index)}

    # Convert metrics_dict to DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # Drop any rows with NaN metrics to ensure clean tables
    metrics_df = metrics_df.dropna(subset=['Cumulative Return Mean (%)', 'MDD Mean (%)', 'Sharpe Ratio Mean'])

    # Create summary DataFrame with formatted strings
    metrics_summary_df = metrics_df.copy()
    metrics_summary_df['Cumulative Return (%)'] = metrics_df['Cumulative Return Mean (%)'].round(2).astype(str) + " ± " + metrics_df['Cumulative Return Std (%)'].round(2).astype(str)
    metrics_summary_df['MDD (%)'] = metrics_df['MDD Mean (%)'].round(2).astype(str) + " ± " + metrics_df['MDD Std (%)'].round(2).astype(str)
    metrics_summary_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio Mean'].round(2).astype(str) + " ± " + metrics_df['Sharpe Ratio Std'].round(2).astype(str)
    metrics_summary_df = metrics_summary_df[['Method', 'Cumulative Return (%)', 'MDD (%)', 'Sharpe Ratio']]

    # Print the comparison table
    print(f"\n=== Metrics Comparison for {current_algo.upper()} ===")
    print(metrics_summary_df.to_string(index=False))
    print("\n")

    # Create separate DataFrames for ranking
    ranking_cum_ret = metrics_df[['Method', 'Cumulative Return Mean (%)']].copy()
    ranking_cum_ret = ranking_cum_ret.sort_values(by='Cumulative Return Mean (%)', ascending=False)
    
    ranking_mdd = metrics_df[['Method', 'MDD Mean (%)']].copy()
    ranking_mdd = ranking_mdd.sort_values(by='MDD Mean (%)', ascending=True)  # Lower MDD is better
    
    ranking_sharpe = metrics_df[['Method', 'Sharpe Ratio Mean']].copy()
    ranking_sharpe = ranking_sharpe.sort_values(by='Sharpe Ratio Mean', ascending=False)
    
    # Print rankings
    print(f"=== Rankings for {current_algo.upper()} ===")
    
    print("\nCumulative Return (%):")
    for idx, row in ranking_cum_ret.iterrows():
        print(f"{row['Method']}: {row['Cumulative Return Mean (%)']:.2f}%")
    
    print("\nMaximum Drawdown (MDD %) [Lower is Better]:")
    for idx, row in ranking_mdd.iterrows():
        print(f"{row['Method']}: {row['MDD Mean (%)']:.2f}%")
    
    print("\nSharpe Ratio [Higher is Better]:")
    for idx, row in ranking_sharpe.iterrows():
        print(f"{row['Method']}: {row['Sharpe Ratio Mean']:.2f}")
    
    print("\n")

    # Debugging: Check if all means align with result.index
    for exp_name, stats in experiment_stats.items():
        mean_length = len(stats['mean'])
        result_length = len(result.index)
        if mean_length != result_length:
            print(f"Warning: Mean length for '{exp_name}' ({mean_length}) does not match result index length ({result_length}). Reindexing.")
            experiment_stats[exp_name]['mean'] = stats['mean'].reindex(result.index).fillna(method='ffill')
            experiment_stats[exp_name]['std'] = stats['std'].reindex(result.index).fillna(0)

    # Plotting section
    plt.figure(figsize=(16, 9))  # Increased figure size for better readability
    method_styles = {
    'CQL': {'color': '#1f77b4', 'linestyle': '-'},           # Blue solid
    'IQL': {'color': '#ff7f0e', 'linestyle': '--'},          # Orange dashed
    'BC': {'color': '#2ca02c', 'linestyle': '-.'},           # Green dash-dot
    'DT LoRA GPT2': {'color': '#d62728', 'linestyle': ':'},  # Red dotted
    'DT LoRA Random Weight GPT2': {'color': '#9467bd', 'linestyle': '-'},  # Purple solid
    'A2C': {'color': '#8c564b', 'linestyle': '--'},          # Brown dashed
    'DDPG': {'color': '#e377c2', 'linestyle': '-'},          # Pink solid
    'PPO': {'color': '#7f7f7f', 'linestyle': '-'},           # Gray solid
    'TD3': {'color': '#bcbd22', 'linestyle': '--'},          # Olive dashed
    'SAC': {'color': '#17becf', 'linestyle': '-'},           # Cyan solid
    'DJIA': {'color': '#000000', 'linestyle': '-'},          # Black solid
    # Add more methods here if needed
}
    # Plot DJIA
    plt.plot(result.index, result['DJIA'], label="Dow Jones Index", linestyle=method_styles['DJIA']['linestyle'], color=method_styles['DJIA']['color'])

    # Define color palette and line styles
    color_palette = plt.get_cmap('tab10').colors  # Colorblind-friendly palette
    line_styles = ['-', '--', '-.', ':']  # Different line styles

    # Plot experiment groups
    for idx, (exp_name, stats) in enumerate(experiment_stats.items()):
        mean = stats['mean']
        std = stats['std']
        
        # Ensure mean and std are aligned with result.index
        mean = mean.reindex(result.index).fillna(method='ffill')
        std = std.reindex(result.index).fillna(0)

        # Assign colors and line styles
        # color = color_palette[idx % len(color_palette)]
        # linestyle = line_styles[idx % len(line_styles)]

        def exp_name_formatter(exp_name):
            exp_names = exp_name.split('_')
            if len(exp_names) == 1:
                return exp_name
            elif len(exp_names) == 2:
                return exp_names[1].upper()
            elif len(exp_names) == 3:
                return None
            elif len(exp_names) == 4:
                return exp_names[1].upper() + ' LoRA ' + 'GPT2'
            elif len(exp_names) == 6:
                return exp_names[1].upper() + ' LoRA ' + 'Random Weight ' + 'GPT2'
            else:
                return exp_name

        # Plot mean
        line, = plt.plot(result.index, mean, label=exp_name_formatter(exp_name), linestyle=method_styles[exp_name_formatter(exp_name)]['linestyle'], color=method_styles[exp_name_formatter(exp_name)]['color'])
        
        # Plot error bandsy (mean ± 1 std)
        plt.fill_between(result.index, mean - std, mean + std, color=method_styles[exp_name_formatter(exp_name)]['color'], alpha=0.2)

    # Set title and labels with enhanced formatting
    plt.title(f"Performance Comparison Under {current_algo.upper()} Expert Agent", fontsize=20, fontweight='bold')
    plt.xlabel("Date", fontsize=16, fontweight='bold')
    plt.ylabel("Total Asset Value ($)", fontsize=16, fontweight='bold')

    # import matplotlib.dates as mdates
    plt.xticks(result.index[0::30])
    # Add 'Test Phase' annotation with date range
    plt.text(0.5, 0.95, 'Test Phase: July 1, 2020 - October 29, 2021', 
             transform=plt.gca().transAxes, fontsize=14, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    # After all lines are plotted, sort the legend alphabetically
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda t: t[0].lower())  # Sort alphabetically, case-insensitive
    sorted_labels, sorted_handles = zip(*sorted_pairs)

    # Position legend outside the plot with sorted items
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

    # Enhance layout and aesthetics
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save the plot with an informative filename
    plt.savefig(f'performance_comparison_DT-LoRA-GPT2_{current_algo.upper()}_Expert.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Results saved as 'performance_comparison_DT-LoRA-GPT2_{current_algo.upper()}_Expert.png'")
    print("----------------------------------------")
