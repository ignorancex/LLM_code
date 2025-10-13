
import pandas as pd

def read_data_common(df_list: (list[pd.DataFrame] | pd.DataFrame)) -> pd.DataFrame:
    if isinstance(df_list, list):
        df_val = pd.concat(df_list, axis=0).reset_index(drop=True)
    else:
        df_val = df_list
    df_val.rename(columns={'IntelliValSet_Raw|en': 'L'}, inplace=True)
    return df_val

def _read_data_1009():
    df1 = pd.read_csv('data/1009_new_log_eq_L_N_D.csv', dtype={'N': float, 'D': float})
    df2 = pd.read_csv('data/1009_new_log_eq_L_N_D_val_bpc.csv')
    df_val = df2.merge(df1.copy(), on='exp_name', how='inner')
    return df_val

def read_data_1009():
    df_val = _read_data_1009()
    return read_data_common(df_val)

def _read_data_1011():
    df3 = pd.read_csv('data/1009_new_log_eq_L_N_D_add_1011.csv')
    df4 = pd.read_csv('data/1009_new_log_eq_L_N_D_val_bpc_add_1011.csv')
    df_val_2 = df3.merge(df4.copy(), on='exp_name', how='inner')
    return df_val_2

def read_data_1011():
    df_val_1 = _read_data_1009()
    df_val_2 = _read_data_1011()
    return read_data_common([df_val_1, df_val_2])

def _read_data_1030():
    df5 = pd.read_csv('data/1009_new_log_eq_L_N_D_add_1030.csv')
    df6 = pd.read_csv('data/1009_new_log_eq_L_N_D_val_bpc_add_1030.csv')
    df_val_3 = df5.merge(df6.copy(), on='exp_name', how='inner')
    return df_val_3

def read_data_1030():
    df_val_1 = _read_data_1009()
    df_val_2 = _read_data_1011()
    df_val_3 = _read_data_1030()
    return read_data_common([df_val_1, df_val_2, df_val_3])

def _read_data_1115():
    df1 = pd.read_csv('data/1009_new_log_eq_L_N_D_add_1115.csv', dtype={'N': float, 'D': float})
    df2 = pd.read_csv('data/1009_new_log_eq_L_N_D_val_bpc_add_1115.csv')
    df_val = df2.merge(df1.copy(), on='exp_name', how='inner')
    return df_val

def read_data_1115():
    df_val_1 = _read_data_1009()
    df_val_2 = _read_data_1011()
    df_val_3 = _read_data_1030()
    df_val_4 = _read_data_1115()
    return read_data_common([df_val_1, df_val_2, df_val_3, df_val_4])

def _read_data_1222():
    df5 = pd.read_csv('data/1009_new_log_eq_L_N_D_add_1222.csv')
    df6 = pd.read_csv('data/1009_new_log_eq_L_N_D_val_bpc_add_1222.csv')
    df_val_3 = df5.merge(df6.copy(), on='exp_name', how='inner')
    return df_val_3

def _read_data_1231():
    df5 = pd.read_csv('data/1009_new_log_eq_L_N_D_add_1231.csv')
    df6 = pd.read_csv('data/1009_new_log_eq_L_N_D_val_bpc_add_1231.csv')
    df_val_3 = df5.merge(df6.copy(), on='exp_name', how='inner')
    return df_val_3

def read_data_bilingual():
    df = pd.read_csv('data/bilingual_full.csv', dtype={'N': float, 'D': float})
    df_bpc = pd.read_csv('data/bilingual_bpc.csv')
    df_val = df_bpc.merge(df.copy(), on='exp_name', how='inner')
    df_val.rename(columns={'IntelliValSet_Raw|en': 'L'}, inplace=True)
    return df_val

def read_data_code():
    df_val = pd.read_csv('data/codescaling_results_new.csv', dtype={'N': float, 'D': float})
    df_val.rename(columns={'IntelliValSet_code': 'L'}, inplace=True)
    return df_val

def read_data_math():
    df_val = pd.read_csv('data/math_140_merge.csv', dtype={'N': float, 'D': float})
    df_val.rename(columns={'IntelliValSet_code': 'L2'}, inplace=True)
    df_val.rename(columns={'IntelliValSet_math_V2': 'L'}, inplace=True)
    return df_val

def read_data_val_big_d():
    df_val = pd.read_csv('data/val_big_d_full.csv', dtype={'N': float, 'D': float})
    df_val.rename(columns={'IntelliValSet_Raw|en': 'L'}, inplace=True)
    return df_val

def read_data_val_big_d_smooth_loss():
    df_val = read_data_val_big_d()
    df_val.rename(columns={'L': 'L_val_big_d'}, inplace=True)
    df_val.rename(columns={'smooth loss': 'L'}, inplace=True)
    return df_val

def read_data_rope():
    df_val = pd.read_csv('data/rope.csv', dtype={'N': float, 'D': float})
    df_val.rename(columns={'IntelliValSet_Raw|en': 'L'}, inplace=True)
    df_val = df_val[df_val['L'].notna()]
    return df_val

def read_data_lr_bs():
    df_val = pd.read_csv('data/lr_bs_full.csv', dtype={'N': float, 'D': float})
    df_val.rename(columns={'IntelliValSet_Raw|en': 'L'}, inplace=True)
    return df_val

def read_data_1222():
    df_val_1 = _read_data_1009()
    df_val_2 = _read_data_1011()
    df_val_3 = _read_data_1030()
    df_val_4 = _read_data_1222()
    df_val_5 = _read_data_1231()
    df_val = read_data_common([df_val_1, df_val_2, df_val_3, df_val_4, df_val_5])
    df_val['round(D/N)'] = df_val['new_round(D/N)']
    df_val['D/N'] = df_val['new_D/N']
    df_val['ND'] = df_val['new_ND']
    df_val['N'] = df_val['exact_N']
    return df_val

def read_data_1222_smooth_loss():
    df_val = read_data_1222()
    df_val.rename(columns={'L': 'L_1222'}, inplace=True)
    df_val.rename(columns={'smooth loss': 'L'}, inplace=True)
    return df_val

def compare_L_values():
    df_1222 = read_data_1222()
    df_rope = read_data_rope()
    df_1222 = df_1222[['N', 'D', 'L']]
    df_rope = df_rope[['N', 'D', 'L']]
    df_rope.rename(columns={'L': 'L_rope'}, inplace=True)
    df_1222 = df_1222.merge(df_rope, on=['N', 'D'], how='inner')
    df_1222['diff'] = (df_1222['L'] - df_1222['L_rope'])
    df_1222['diff_abs'] = df_1222['diff'].abs()
    df_1222 = df_1222.sort_values(by='diff_abs', ascending=False)
    return df_1222
if (__name__ == '__main__'):
    df_val = read_data_1222()
    df_val.to_csv('data/1222_full.csv', index=False)
