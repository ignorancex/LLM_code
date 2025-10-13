import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

venues = ['ACL', 'EMNLP', 'EMSE', 'FORGE', 'ICPC', 'ISSTA', 'KDD', 'MSR', 'NAACL', 'NeurIPS', 'SANER', 
 'arXiv', 'TSE', 'FSE', 'ICSE', 'TOSEM', 'ASE']
paper_counts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4]

df = pd.DataFrame({'Venue': venues, 'Number of Papers': paper_counts})

df_reversed = df[::-1]
colors = ['#E57373', '#4DB6AC', '#1976D2', '#FBC02D', '#C2185B', '#FF9800', '#BDBDBD', '#42A5F5', 
          '#FFAB91', '#81C784', '#1E88E5', '#F44336', '#B2EBF2', '#424242', '#9575CD', '#FFD600', 
          '#7E57C2', '#212121']
colors_reversed = colors[::-1] 

fig, ax = plt.subplots(figsize=(7, 4))

ax.barh(df_reversed['Venue'], df_reversed['Number of Papers'], color=colors_reversed, edgecolor='black')

ax.set_xlim(0, max(df_reversed["Number of Papers"]))

ax.xaxis.set_major_locator(MaxNLocator(integer=True))


ax.set_xlabel("Number of Papers", fontsize=10)
ax.set_ylabel("Venue")
ax.tick_params(axis='both', which='major', labelsize=10)

ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_ylim(-0.5, len(df_reversed) - 0.5) 

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig("../visualizations/venue_barplot.png", format="png", dpi=300, bbox_inches="tight")

plt.show()
