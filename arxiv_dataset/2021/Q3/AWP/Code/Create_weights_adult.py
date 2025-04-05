import pandas as pd
from collections import Counter
import sys

input_string = list(sys.argv)
data = pd.read_csv(input_string[1])
N = int(input_string[2])  # N value as defined in paper
weightParameter = input_string[3]  # The parameter the weights will be created according to. see paper for more detail
values = list(data[weightParameter].unique())
for val in values:  # removes rare values - gives their examples a weight of 0
    if Counter(data[weightParameter])[val] < len(data) * 0.05:
        values.remove(val)
weights = []
for row in range(len(data)):
    if data[weightParameter][row] in values:
        weights.append(pow(N, values.index(data[weightParameter][row])))
    else:
        weights.append(0)

# normalizing:
df = pd.DataFrame(weights, columns=['Weight'])
df = df.multiply(other=1 / df['Weight'].sum())

# checking weight distribution among examples is valid
check_sum = 0
count = 0
m = 0
for a in range(len(df['Weight'])):
    check_sum += df.iloc[a, 0]
    if df.iloc[a, 0] > 0:
        count += 1
        if df.iloc[a, 0] > m:
            m = df.iloc[a, 0]
print("Sum weights value is:" + str(check_sum))
print("Number of examples with nonzero weight is " + str(count))
print("Maximum weight for example is " + str(m))

df.to_csv(input_string[4])
