import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import tasks, tasks_new

pd.set_option('display.max_columns', None)

model_names = ['FCN', 'InceptionTime']
xai_names = ['Saliency']

num_tasks = len(tasks) + len(tasks_new)

result = pd.read_csv(f'output/half_dataset_clsterting2dTrue_test_new.csv')
# result = pd.read_csv(f'output/half_dataset_test.csv')

baseline = result[result['xai_name'].isnull()]
df = result[~result['xai_name'].isnull()]
df = df.merge(baseline, on=['model_name', 'task_name'], how='left')
df.loc[df['task_name'] == 'DistalPhalanxOutlineCorrect', 'task_name'] = 'DPOC'
df['acc_drop'] = df['acc_score_y'] - df['acc_score_x']

df['tag'] = df[['implet_src_x', 'mode_x', 'method_x']].agg(', '.join, axis=1)

plt.figure(dpi=300, figsize=(8, 8))

for model_name in model_names:
    for xai_name in xai_names:
        ax = sns.barplot(df[(df['model_name'] == model_name) & (df['xai_name_x'] == xai_name)],
                    x='task_name', y='acc_drop', hue='tag')

        bars = ax.patches  # Get all bars
        for i, bar in enumerate(bars):
            if (i // num_tasks) % 2 == 1:  # Apply to odd-numbered bars in each group
                bar.set_edgecolor(plt.get_cmap('tab10')((i // num_tasks) // 2))
                bar.set_facecolor('none')
                bar.set_hatch('//')
            else:
                bar.set_facecolor(plt.get_cmap('tab10')((i // num_tasks) // 2))

        # plt.ylim(-0.105, 0.505)
        plt.title(f'{model_name}, {xai_name}')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'figure/half_dataset/{model_name}_{xai_name}_clsterting2dTrue.png')
        plt.clf()


