import datasets
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
class ClaraAnalysis():
    def __init__(self, dataset_path=None) -> None:
        self.techniques = {0: 'CLARA', 1: 'FA(L)', 3: 'FA(L+E)', 4: 'SARFGEN'}
        self.difficulty = {
            '1A': 1000,
            '4A': 800,
            '4C': 1300,
            '50A': 800,
            '208A': 900,
            '214A': 800,
            '255A': 800,
            '265A': 800,
            '492B': 1200,
            '510A': 800,
            '1097A':800, 
            '1360B':800, 
            '1363A':1200, 
            '1364A':1200, 
            '1369B':1200, 
            '1370A':800, 
            '1382B':1100, 
            '1385A':800, 
            '1391A':800, 
            '1391B':800
        }
        if not dataset_path:
            self.dataset_path = 'dataset'
        else:
            self.dataset_path = dataset_path
        self.dataset = datasets.load_from_disk(self.dataset_path)
        self.dataset = self.dataset.map(self.assign_technique)
        self.dataset = self.dataset.map(self.parse_error_output)
        self.dataframe = self.dataset['final'].to_pandas()   
        # Setting proper types of data  
        self.dataframe['Technique'] = pd.Categorical(self.dataframe['Technique'])
        self.dataframe['Percentage Repaired'] = self.dataframe['Percentage Repaired'].astype(float)
        print("Test Available")
        print(self.dataframe.groupby(['Technique', 'Test Available'])['Technique'].count())
        self.dataframe = self.dataframe[self.dataframe['Test Available'].isna()!=True]
        print("List of Parse Errors:")
        print(self.dataframe[self.dataframe['Technique']=='CLARA'].groupby(['Parse Error Output'])['Technique'].count())
        self.dataframe = self.dataframe[self.dataframe['Parse Error'] != 'True']
        self.dataframe = self.dataframe[self.dataframe['Test Available'] == 'Yes']
        timeout = self.dataframe[self.dataframe['Timeout']==True]
        print("List of Timeouts")
        print(timeout.groupby(['Technique'])['Technique'].count())
        self.dataframe = self.dataframe[self.dataframe['Timeout'] !=True]
        self.dataframe['ci'] = self.dataframe['Correct File'] + '_' + self.dataframe['Incorrect File']
        grouped = self.dataframe.groupby('ci')
        self.dataframe = grouped.filter(lambda x: x['Technique'].count() == 3)
        self.dataframe['Difficulty'] = self.dataframe['Problem'].map(self.difficulty)


    def parse_error_output(self, x):
        if x['Parse Error'] == 'True':
            return {'Parse Error Output': re.findall(r':([^\(]+)', x['Error Output'].split('\n')[-2])[0]}
        else:
            return {'Parse Error Output': None}

    def assign_technique(self, x):
        x['Technique'] = self.techniques[x['Technique']]
        return x
    def group_analyze(self, column, type=None, filter=None):
        if type is not None:
          self.dataframe[column] = self.dataframe[column].astype(type)
        return self.dataframe.groupby([column])[column].count()

    def summary_statistic(self):
        pass

    def groups_analyze(self, columns, df=None):
        if df is None:
            df = self.dataframe
        return df.groupby(columns)[columns].count()

    def filter_group_analyze(self, column, type=None):
        dt_parse_error = self.dataset.filter(lambda x: x['Parse Error'] == 'True')\
          .map(self.parse_error_output)
        print(self.group_analyze(dt_parse_error['final'].to_pandas(), 'Parse Error Output', type="category"))

    def draw_boxplot(self, groups, title='Title', ylabel='YLabel'):
        data = [self.dataframe[grp].astype(float).to_numpy() for grp in groups]
        sns.boxplot(data)


class ClaraHelperMethods:
    def __init__(self):
        self.figsize = (10,7)
        self.axes = [0,0,1,1]
        self.font_scale = 1
        self.style = "white"
        self.palette = {'CLARA':"#9b59b6", 'FA(L)':"#3498db", 'FA(L+E)':"#95a5a6", "SARFGEN": "#D5B339"}
        self.other_palette = ["#9b59b6","#3498db","#95a5a6"]
        self.order = [
            "4A",
            "50A",
            "214A",
            "255A",
            "265A",
            "510A",
            "1097A",
            "1360B",
            "1370A",
            "1385A",
            "1391A",
            "1391B",
            "208A",
            "1A",
            "1382B",
            "492B",
            "1363A",
            "1364A",
            "1369B",
            "4C"
        ]
        self.fontsize = 36
        self.legend = 'upper right'
    def add_bin_col(self, x, df, col='locs_bin'):
        row = df.loc[(df[col].isna() != True)]
        row = row[row['Correct File'] == x['Correct File']]
        if len(row[col]) > 0:
            x[col] = row[col].iloc[0]
        return x
    
    def do_bin(self, df, new_col='locs_bin', bin_column='Correct Locs', bins=[0,10,20,30,40,50,60,70,80]):
        df[new_col] = pd.cut(np.array(df[bin_column]), bins=bins)
        # df[df[new_col].isna() != True]['Correct File']
        df = df.apply(lambda x: self.add_bin_col(x, df, col=new_col), axis=1)
        return df
    def successful_repair_percentage(self, df, col=['Technique']):
        succesful_repairs = df[df['Repairs'].isna()!=True].groupby(col)['Problem'].count()
        total_repairs = df.groupby(col)['Problem'].count()
        repaired = succesful_repairs/total_repairs
        return repaired
    
       
          
    def plot_success(self, df, figname, xlabel, ylabel="",
                     y_labels=None, x_labels=None, legend=None,
                     repair=None, col=['Technique'],
                     figsize=None, axes=None, font_scale=None,
                     style=None, fontsize=None, order=None, hue_order=None):
        figsize = figsize if figsize else self.figsize
        axes = axes if axes else self.axes
        font_scale = font_scale if font_scale else self.font_scale
        style = style if style else self.style
        fontsize = fontsize if fontsize else self.fontsize
        if not repair:
            repair = self.successful_repair_percentage(df, col=col)
            display(repair)
        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes(axes)
        sns.set(font_scale=font_scale)
        sns.set_style(style)
        x = repair.index.get_level_values(col[0]) if len(col)>1 else repair.index
        y = repair.values*100
        hue = repair.index.get_level_values(col[1]) if len(col)>1 else None
        sns.barplot(x=x,
                    hue=hue, 
                    y=y, 
                    ax=ax, 
                    palette=self.palette,
                    order=order,
                    hue_order=hue_order)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if legend:
            ax.legend(loc=legend, bbox_to_anchor=(1.3, 1), borderaxespad=0, fontsize=24)
        else:
            ax.legend([],[],frameon=False)
        if x_labels:
            ax.set_xticklabels(x_labels, size = fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), size = fontsize)
        if y_labels:
            ax.set_yticklabels(y_labels, size = fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(), size = fontsize)
        plt.savefig(figname, format="pdf", bbox_inches='tight')
        
    def plot_box(self, df, figname, xlabel, ylabel="",
                 orient="v",col=['Correct Locs', 'Technique'],
                 x_labels=None, y_labels=None,order=None,
                 figsize=None, hue=None, whis=1.5, legend=None,
                 axes=None, font_scale=None,style=None, fontsize=None, 
                 hue_order=None):
        figsize = figsize if figsize else self.figsize
        axes = axes if axes else self.axes
        font_scale = font_scale if font_scale else self.font_scale
        style = style if style else self.style
        fontsize = fontsize if fontsize else self.fontsize
        fig = plt.figure(figsize = self.figsize)
        palette = self.palette
        if not hue:
            palette = self.other_palette
        if not hue_order:
            hue_order = self.palette.keys()
        ax = fig.add_axes(self.axes)
        if not order:
            order = df.groupby(col[1])[col[1]].count().index
        ax = sns.boxplot(
            data=df, 
            y=col[0],
            x=col[1],
            hue=hue,
            orient=orient,
            whis=whis,
            order=order,
            hue_order=hue_order,
            palette=palette)
        
        if legend:
            ax.legend(loc=legend, bbox_to_anchor=(1.3, 1), borderaxespad=0, fontsize=24)
        else:
            ax.legend([],[],frameon=False)
        if x_labels:
            ax.set_xticklabels(x_labels, size = fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), size = fontsize)
        if y_labels:
            ax.set_yticklabels(y_labels, size = fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(), size = fontsize)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.savefig(figname, format="pdf", bbox_inches='tight')
    
    def plot_bar(self, df, figname, xlabel, ylabel="",
                 orient="v",col=['Correct Locs', 'Technique'],
                 x_labels=None, y_labels=None,order=None,
                 figsize=None, hue=None, whis=1.5, legend=None,
                 axes=None, font_scale=None,style=None, fontsize=None):
        figsize = figsize if figsize else self.figsize
        axes = axes if axes else self.axes
        font_scale = font_scale if font_scale else self.font_scale
        style = style if style else self.style
        fontsize = fontsize if fontsize else self.fontsize
        fig = plt.figure(figsize = self.figsize)
        palette = self.palette
        if not hue:
            palette = self.other_palette
        ax = fig.add_axes(self.axes)
        if not order:
            order = df.groupby(col[1])[col[1]].count().index
#         ax = sns.boxplot(
#             data=df, 
#             y=col[0],
#             x=col[1],
#             hue=hue,
#             orient=orient,
#             whis=whis,
#             order=order,
#             palette=palette)
        ax = sns.barplot(
            data=df,
            hue=hue, 
            y=col[0],
            x=col[1],
            order=order,
            ax=ax, 
            palette=self.palette)
        if legend:
            ax.legend(loc=legend, bbox_to_anchor=(1.3, 1), borderaxespad=0, fontsize=24)
        else:
            ax.legend([],[],frameon=False)
        if x_labels:
            ax.set_xticklabels(x_labels, size = fontsize)
        if y_labels:
            ax.set_yticklabels(y_labels, size = fontsize)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.savefig(figname, format="pdf", bbox_inches='tight')