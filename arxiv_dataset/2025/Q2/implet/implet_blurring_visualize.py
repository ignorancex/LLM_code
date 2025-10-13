import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import tasks, tasks_new

pd.set_option('display.max_columns', None)

# modes = ['single', 'all', 'single_pos_only', 'all_pos_only']
mode = 'single'

model_names = ['FCN', 'InceptionTime']
xai_names = ['GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion', 'Saliency']
tasks = sorted(tasks + tasks_new)
print(tasks)
result_imp = pd.read_csv(f'output/blurring_test_{mode}_new.csv')
result_st = pd.read_csv(f'output/blurring_test_ST.csv')
result_imp.replace('repl_random_loc', 'repl_random_loc_implet', inplace=True)
result_st.replace('repl_implet', 'repl_shapelet', inplace=True)
result_st.replace('repl_random_loc', 'repl_random_loc_shapelet', inplace=True)

baseline = result_imp[result_imp['xai_name'].isnull()]

df_imp = result_imp[~result_imp['xai_name'].isnull()]
df_st = result_st[~result_st['xai_name'].isnull()]
df = pd.concat([df_imp, df_st])
df.replace('DistalPhalanxOutlineCorrect', 'DPOC', inplace=True)
df.replace('GunPointMaleVersusFemale', 'GunPointGender', inplace=True)

df = df.merge(baseline, on=['model_name', 'task_name'], how='left')

df['acc_drop'] = df['acc_score_y'] - df['acc_score_x']

ylim = [df['acc_drop'].min(), df['acc_drop'].max()]
ylim = [ylim[0] - 0.05, ylim[1] + 0.05]
print('ylim =', ylim)

plt.figure(dpi=300)

for model_name in model_names:
    for xai_name in xai_names:
        idx = (df['model_name'] == model_name) & (
                (df['xai_name_x'] == xai_name) | (df['xai_name_x'] == 'ShapeletTransform')
        )
        sns.barplot(df[idx],
                    x='task_name', y='acc_drop', hue='method_x')
        plt.ylim(*ylim)
        plt.title(f'{model_name}, {xai_name}')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'figure/blurring_test_{mode}_new/{model_name}_{xai_name}.png')
        plt.clf()

# for model_name in model_names:
#     for task in tasks:
#         sns.barplot(df[(df['model_name'] == model_name) & (df['task_name'] == task)],
#                     x='xai_name_x', y='acc_drop', hue='method_x')
#         plt.ylim(-0.105, 0.505)
#         plt.title(f'{model_name}, {task}')
#         plt.xticks(rotation=70)
#         plt.tight_layout()
#         plt.savefig(f'figure/blurring_test_{mode}/byTask/{model_name}_{task}.png')
#         plt.clf()
