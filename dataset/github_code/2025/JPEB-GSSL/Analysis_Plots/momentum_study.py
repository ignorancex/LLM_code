import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Momentum': [0, 0.8, 0.9, 0.99, 0.999, 1],
    'Accuracy': [76.46, 77.00, 77.00, 77.01, 74.93, 70.73]
}


df = pd.DataFrame(data)

df['Momentum_Index'] = range(len(df['Momentum']))


plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='Momentum_Index', y='Accuracy', marker='o')

plt.xticks(ticks=df['Momentum_Index'], labels=df['Momentum'], fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Momentum', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True)

plt.show()
