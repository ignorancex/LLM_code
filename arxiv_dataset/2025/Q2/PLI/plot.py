import matplotlib.pyplot as plt
import pandas as pd
from math import pi

def plot_polar():
    # 绘制雷达图
    # Create a DataFrame
    # \begin{tabular}{lccccc}
    # \toprule
    #        & FashionIQ & CIRR  & CIRCO & GeneCIS \\
    # Method & R@10      & R@10  & mAP@5 & R@3     & Average\\
    # \midrule
    # LinCIR \cite{gu2023languageonly} & 25.00     & 65.35 & \textbf{12.75} & 29.65  & 33.19 \\
    # Ours   & \textbf{34.10} & \textbf{73.30} & {10.03}   &  \textbf{37.65} & \textbf{38.77} \\
    # % Ours (MSCOCO) & \textbf{36.53} & \underline{70.77} &  6.60 & \underline{34.92} & \underline{37.21}\\
    # \bottomrule
    # \end{tabular}
    data = pd.DataFrame({
        'group': ['LinCIR', 'Ours'],
        'FashionIQ': [25.00, 34.10],
        'CIRR': [65.35, 73.30],
        'CIRCO': [12.75, 10.03],
        'GeneCIS': [29.65, 37.65],
        'Average': [33.19, 38.77]
    })

    # number of variable
    categories = list(data)[1:]
    N = len(categories)

    # 设置每个点的角度值
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # 初始化极坐标网格
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    # 设置角度偏移
    ax.set_theta_offset(pi / 2)
    # 设置顺时针还是逆时针，1或者-1
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    # Draw ylabels
    ax.set_rlabel_position(0)
    # 每个变量的范围不同：fashionIQ 20~40, CIRR 60~80, CIRCO 10~15, GeneCIS 25~40, Average 30~40
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
    plt.ylim(0, 100)

    # Plot data
    # LinCIR
    values = data.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="LinCIR")
    ax.fill(angles, values, 'b', alpha=0.1)
    # Ours
    values = data.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Ours")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the plot
    plt.show()

plot_polar()


