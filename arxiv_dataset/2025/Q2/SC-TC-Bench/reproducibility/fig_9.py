import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('source_data/regional_name_and_characteristics.csv')
    df_mc = df[df['region']=="Mainland China"]
    df_tw = df[df['region']=="Taiwan"]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Plot KDE
    plt.figure(figsize=(8, 7))
    sns.kdeplot(df_mc['count'].tolist(), fill=True, bw_adjust=0.5, linewidth=2)
    plt.xlabel("Population")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("figure/regional_name/density_name_mainland_china.pdf")

    plt.figure(figsize=(8, 7))
    sns.kdeplot(df_tw['count'].tolist(), fill=True, bw_adjust=0.5, linewidth=2)
    plt.xlabel("Population")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("figure/regional_name/density_name_taiwan_region.pdf")

    