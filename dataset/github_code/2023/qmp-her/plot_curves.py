import seaborn as sbn
import matplotlib.pyplot as plt
import pandas as pd
progress = pd.read_csv('/home/francisco/primitives-her/log/primitives_try-ddpg-FetchPush-v1-maxq/progress.csv')["Success"]
progress2 =pd.read_csv('/home/francisco/primitives-her/log/normal_try-ddpg-FetchPush-v1-normal/progress.csv')["Success"]
data = pd.concat([progress, progress2, progress3], axis=1)
data.columns.values[0] = "Ours (2 primitives)"
data.columns.values[1] = "Ours (7 primitives)"
sbn.set()
plt.plot()
plt.xlabel("Policy Updates")
plt.ylabel("Success Rate")
plt.ylim([0,1])
ax = sbn.lineplot(data=data)
plt.savefig('progress_pick.png')

