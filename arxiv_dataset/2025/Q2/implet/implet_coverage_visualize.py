import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


model_names = ['FCN', 'InceptionTime']
xai_names = ['DeepLift', 'GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion',
             'Saliency']
tasks = ['GunPoint', "ECG200", "DPOC", "PowerCons", "Earthquakes", "Strawberry"]
result = pd.read_csv(f'output/implet_coverage.csv')

# baseline = result[result['xai_name'].isnull()]
df = result[~result['xai_name'].isnull()]
# df = df.merge(baseline, on=['model_name', 'task_name'], how='left')
df.loc[df['task_name'] == 'DistalPhalanxOutlineCorrect', 'task_name'] = 'DPOC'
# df['acc_drop'] = df['acc_score_y'] - df['acc_score_x']
print(df.columns)
plt.figure(dpi=300)

for model_name in model_names:
    for xai_name in xai_names:
        sns.barplot(df[(df['model_name'] == model_name) & (df['xai_name'] == xai_name)],
                    x='task_name', y='total_coverage')
        plt.ylim(-0.05, 1.005)
        plt.title(f'{model_name}, {xai_name}')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'figure/implet_coverage/{model_name}_{xai_name}.png')
        plt.clf()

for model_name in model_names:
    for task in tasks:
        sns.barplot(df[(df['model_name'] == model_name) & (df['task_name'] == task)],
                    x='xai_name', y='total_coverage')
        plt.ylim(-0.05, 1.005)
        plt.title(f'{model_name}, {task}')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'figure/implet_coverage/byTask/{model_name}_{task}.png')
        plt.clf()