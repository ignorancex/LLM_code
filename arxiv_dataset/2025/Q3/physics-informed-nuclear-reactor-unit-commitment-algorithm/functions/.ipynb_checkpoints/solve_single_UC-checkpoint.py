import numpy as np
import pandas as pd
import papermill as pm
import os

#define function for tandem run while handeling all required I/O 


def solve_single_UC(model_name, horizon, load_percentage, PHS_percentage, PHS_duration, n, nuclear_unit,\
                 remaining_downtime, kinfTable_udt, commit_at_len_T, D_points, first_run, pmin, VRE_percentage, cost_frac,\
                   mAP1000, mAP300, mAP100, Wind_Cap, Solar_Cap, Nuclear_Cap, PHS_Cap, refuel_span):
    """
    run_instance is a function defined to take all necessary inputs and run UC.ipynb and 
    generate inputs in the correct format.

    [SC: FILL THE DEFINITIONS]
    Parameters:
    -model_name:
    -horizon:
    -load_percentage:
    -PHS_percentage:
    -PHS_percentage: 
    -PHS_duration:
    -net_cap:
    -n:
    -nuclear_unit:
    -remaining_downtime:
    -kinfTable_udt:
    -commit_at_len_T:
    -D_points:
    -first_run:

    Returns:
    -transposed_df:
    -shut_df:
    -remaining_downtime:
    -commit_result:
    -kinfTable_udt:
    -commit_at_len_T:
    -D_points:
    
    """
    path=os.getcwd()
    output_path_for_this_run = f"output_notebook_{n}.ipynb"
    pm.execute_notebook(
        path + "/UC.ipynb",
        # path + "/UC.py",
        output_path_for_this_run,
        parameters={'model_name': model_name, 'horizon':horizon,'load_percentage':load_percentage,'PHS_percentage': PHS_percentage,\
                    'PHS_duration':PHS_duration,'n': n, 'nuclear_unit': nuclear_unit, 'remaining_downtime': remaining_downtime.to_dict(),\
                    'kinfTable_udt': kinfTable_udt.to_dict(),'commit_at_len_T': commit_at_len_T.to_dict(),\
                    'D_points':D_points.to_dict(), 'first_run': first_run, 'pmin': pmin, 'VRE_percentage': VRE_percentage, 'cost_frac': cost_frac, 'mAP1000':mAP1000,'mAP300':mAP300, 'mAP100':mAP100, 'Wind_Cap': Wind_Cap, 'Solar_Cap': Solar_Cap, 'Nuclear_Cap': Nuclear_Cap, 'PHS_Cap': PHS_Cap, 'refuel_span':refuel_span}
    )
    kinfTable_udt=pd.read_csv(f'kinfTable_udt_{n}.csv', index_col=0)
    transposed_df = pd.read_csv(f'transposed_df_{n}.csv', index_col=0)
    shut_df = pd.read_csv(f'shut_df_{n}.csv', index_col=0)
    remaining_downtime = pd.read_csv(f'remaining_downtime_{n}.csv', index_col=0)
    commit_at_len_T = pd.read_csv(f'commit_at_len_T_{n}.csv', index_col=0)
    D_points = pd.read_csv(f'D_points_{n}.csv', index_col=0)
    COMMIT_result=pd.read_csv(f'COMMIT_{n}.csv', index_col=0)
    GEN_result=pd.read_csv(f'GEN_{n}.csv', index_col=0)
    SHUT_result=pd.read_csv(f'SHUT_{n}.csv', index_col=0)
    START_result=pd.read_csv(f'START_{n}.csv', index_col=0)


    os.remove(f'kinfTable_udt_{n}.csv')
    os.remove(f'terminate_loop_indicator_{n}.csv')
    os.remove(f'transposed_df_{n}.csv')
    os.remove(f'COMMIT_{n}.csv')
    os.remove(f'shut_df_{n}.csv')
    os.remove(f'remaining_downtime_{n}.csv')
    os.remove(f'commit_at_len_T_{n}.csv')
    os.remove(f'D_points_{n}.csv')
    os.remove(f'GEN_{n}.csv')
    os.remove(f'SHUT_{n}.csv')
    os.remove(f'START_{n}.csv')
    os.remove(output_path_for_this_run)

    return transposed_df, shut_df, remaining_downtime, COMMIT_result,\
    kinfTable_udt, commit_at_len_T, D_points, GEN_result, SHUT_result, START_result