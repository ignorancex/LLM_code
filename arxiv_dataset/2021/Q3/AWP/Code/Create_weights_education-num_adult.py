import pandas as pd
from collections import Counter
import sys

input_string = list(sys.argv)
data = pd.read_csv(input_string[1])
weight_col = 'education-num'
weights = []
N = int(input_string[2])
for row in range(len(data)):
    if data[weight_col][row] <= 8:
        weights.append(pow(N, 0))
    elif data[weight_col][row] <= 14:
        weights.append(pow(N, data[weight_col][row] - 8))
    else:
        weights.append(pow(N, 14 - 8))

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

df.to_csv(input_string[3])
