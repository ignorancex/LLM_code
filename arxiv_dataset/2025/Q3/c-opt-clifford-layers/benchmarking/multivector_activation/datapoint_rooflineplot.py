import pandas as pd

# ---- Constants ---- #
FREQUENCY_GHZ = 3.6  # CPU clock speed
BYTES_PER_FLOAT = 4

# ---- Memory model ---- #
def bytes_loaded(row):
    B, C, I, K, mode = row['B'], row['C'], row['I'], row['K'], row['Mode']

    if mode == 'Linear':
        return (2*BYTES_PER_FLOAT * B * C * I +  # input + output
                BYTES_PER_FLOAT * C * K +        # weights
                BYTES_PER_FLOAT * C)            # bias             
    else:
        return (2*BYTES_PER_FLOAT * B * C * I # input + output
                )            
# ---- Load data ---- #
csv_path = "multivector_bench_x86_Max.csv"
df = pd.read_csv(csv_path)


# ---- Compute metrics ---- #
df['Total_FLOPs'] = df['FLOP_per_cycle'] * df['Cycles']
df['Bytes_Loaded'] = df.apply(bytes_loaded, axis=1)
df['Operational_Intensity'] = df['Total_FLOPs'] / df['Bytes_Loaded']


# ---- Group by key dimensions ---- #
summary = df.groupby(['Version', 'Mode', 'n', 'I', 'B', 'C', 'K'])[
    ['Operational_Intensity','FLOP_per_cycle']
].mean().reset_index()

# ---- Save and display ---- #
summary.to_csv("operational_intensity_summary.csv", index=False)
print(summary.head())
